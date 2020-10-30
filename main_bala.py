import argparse
import os
import shutil
import time
import tqdm
import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models

import datasets 
import models

from Dataset_h5py import Dataset

from lib.utils import get_factors
from lib.NCEAverage import NCEAverage
from lib.LinearAverage import LinearAverage
from lib.NCECriterion import NCECriterion
from lib.utils import AverageMeter
from test import NN, kNN

from apex.fp16_utils import FP16_Optimizer

import pickle

from PIL import Image

#from create_tsne_plot import compute_tsne
#from find_outlier_images import compute_distribution_of_samples
#from find_outlier_images import compute_knn_distances

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--balanced_sampling',  default=False, action='store_true', help='by default set false, oversamples less populated classes')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-dc', '--data-condition', default=1, type=int, metavar='DC', help='Pick from 1,2,3,4 (default: 1)')
parser.add_argument('-fc', '--future-condition', default=0, type=int, metavar='FC', help='Pick from 0-9 (default: 0)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--fine_tune', default='', type=str, metavar='PATH', help='Fine tune a pre-trained model using new dataset')
parser.add_argument('--iter_size', default=1, type=int, help='caffe style iter size')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--K',  default=200, type=int, help='Default number of neighbors for KNN classification')
parser.add_argument('--low-dim', default=128, type=int, metavar='D', help='feature dimension')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--nce-k', default=4096, type=int, metavar='K', help='number of negative samples for NCE')
parser.add_argument('--nce-m', default=0.5, type=float, help='momentum for non-parametric updates')
parser.add_argument('--nce-t', default=0.07, type=float, metavar='T', help='temperature parameter for softmax')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--recompute-memory', default=False, action='store_true', help='recompute memory on train dataset for evaluation stage')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--static_loss', default=25, type=float, help='set static loss for apex optimizer')
#parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')

best_prec1 = 0
torch.backends.cudnn.enabled = False

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        '''
        print('tensor shape: {}'.format(tensor.shape))
        
        tensor = torch.Tensor(np.array(tensor))
        final_tensor = tensor + torch.randn(tensor.size()) * self.std + self.mean
        
        return Image.fromarray(np.array(final_tensor))
        '''
        tensor_arr = np.array(tensor)
        #print('tensor_arr vals: {} - {}'.format(tensor_arr, np.max(tensor_arr.flatten())))
        transformed_arr = tensor_arr + np.random.random(tensor_arr.shape) * self.std + self.mean
        transformed_arr = transformed_arr.astype(np.uint8)
        #print('transformed_arr vals: {} - {}'.format(transformed_arr.shape, np.max(transformed_arr.flatten())))
        
        
        
        return Image.fromarray(transformed_arr)
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def main():


    global args, best_prec1
    args = parser.parse_args()

    # Initialize distributed processing
    args.distributed = args.world_size > 1
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](low_dim=args.low_dim)

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)


    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet stats
    #                                  std=[0.229, 0.224, 0.225])
    #normalize = transforms.Normalize(mean=[0.234, 0.191, 0.159],  # xView stats
    #                                 std=[0.173, 0.143, 0.127])
    
    #normalize = transforms.Normalize(mean=[163.10509, 90.829124, 45.98254],  # AREDS stats (pre dwg)
    #                                 std=[6.322674, 4.129129, 4.051738])
    normalize = transforms.Normalize(mean=[0.6396272, 0.3561925, 0.18032297], # AREDS stats (pre dwg, post toTensor)
                                     std=[0.0157365, 0.0192954, 0.02142001])
    

    print("Creating datasets")
    
    train_transform = transforms.Compose([transforms.ToPILImage(),
                                          transforms.RandomResizedCrop(224, (.58, 1.0), (1.0, 1.0)),
                                          transforms.ColorJitter(.05, .05, .05, .05), 
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          AddGaussianNoise(0., 2.5),
                                          transforms.ToTensor(),
                                          normalize])
    
    val_transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.ToTensor(), 
                                        normalize])
    
    
    
    print('K: {}'.format(args.K))
    
    train_dir = os.path.join(args.data, 'train/', )
    train_dataset = Dataset(train_dir, transform=train_transform, condition=args.data_condition, fut_cond=args.future_condition)
    
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    elif args.balanced_sampling:

        print("Using balanced sampling")
        # Here's where we compute the weights for WeightedRandomSampler
        class_counts = {v: 0 for v in train_dataset.train_labels} #class_to_idx.values()}
        for path, ndx in train_dataset.samples:
            class_counts[ndx] += 1
        total = float(np.sum([v for v in class_counts.values()]))
        class_probs = [class_counts[ndx] / total for ndx in range(len(class_counts))]

        # make a list of class probabilities corresponding to the entries in train_dataset.samples
        reciprocal_weights = [class_probs[idx] for i, (_, idx) in enumerate(train_dataset.samples)]

        # weights are the reciprocal of the above
        weights = (1 / torch.Tensor(reciprocal_weights))

        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_dataset), replacement=True)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    print("Training on", len(train_dataset), "images. Training batch size:", args.batch_size)

    if len(train_dataset) % args.batch_size != 0:
        print("Warning: batch size doesn't divide the # of training images so ",
              len(train_dataset) % args.batch_size, "images will be skipped per epoch.")
        print("If you don't want to skip images, use a batch size in:", get_factors(len(train_dataset)))

    '''
    val_dataset = datasets.IageFolderInstance(
        valdir,
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]))
    '''
    
    
    val_dir = os.path.join(args.data, 'val/')
    val_dataset = Dataset(val_dir, transform=val_transform, condition=args.data_condition, fut_cond=args.future_condition)
    
    '''
    val_transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    val_dataset = datasets.CIFAR10Instance(root='./data', train=False,
                                           download=True, transform=val_transform)
    '''

    val_bs = args.batch_size #[factor for factor in get_factors(len(val_dataset)) if factor < 300][-1]

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_bs, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print("Validating on", len(val_dataset),  "images. Validation batch size:", val_bs)

    # define lemniscate and loss function (criterion)
    ndata = train_dataset.__len__()
    if args.nce_k > 0:
        lemniscate = NCEAverage(args.low_dim, ndata, args.nce_k, args.nce_t, args.nce_m)
        criterion = NCECriterion(ndata).cuda()
    else:
        lemniscate = LinearAverage(args.low_dim, ndata, args.nce_t, args.nce_m).cuda()
        criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.static_loss, verbose=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            lemniscate = checkpoint['lemniscate']
            print("=> loaded checkpoint '{}' (epoch {}, best_prec1 {})"
                  .format(args.resume, checkpoint['epoch'], checkpoint['best_prec1']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # optionally fine-tune a model trained on a different dataset
    elif args.fine_tune:
        print("=> loading checkpoint '{}'".format(args.fine_tune))
        checkpoint = torch.load(args.fine_tune)
        
        own_state = model.state_dict()
        for name, param in checkpoint['state_dict'].items():
            if name not in own_state or 'fc_new' in name:
                continue
                    
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
        
        #model.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #'''
        own_state = optimizer.state_dict()
        for name, param in checkpoint['optimizer'].items():
            if name not in own_state or 'fc_new' in name:
                continue
                
            #print('own state {} size : {}, param shape: {}'.format(name, own_state[name], param))
                    
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name] = param
            
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.static_loss, verbose=False)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.fine_tune, checkpoint['epoch']))
        #'''
    else:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.static_loss, verbose=False)

    # Optionally recompute memory. If fine-tuning, then we must recompute memory
    if False or args.recompute_memory or args.fine_tune:

        # Aaron - Experiments show that iterating over torch.utils.data.DataLoader will skip the last few
        # unless the batch size evenly divides size of the data set. This shouldn't be the case
        # according to documentation, there's even a flag for drop_last, but it's not working

        # compute a good batch size for re-computing memory
        memory_bs = [factor for factor in get_factors(len(train_loader.dataset)) if factor < 300][-1]
        print("Recomputing memory using", args.data, "with a batch size of", memory_bs)
        #transform_bak = train_loader.dataset.transform
        #train_loader.dataset.transform = val_loader.dataset.transform
        temploader = torch.utils.data.DataLoader(
            train_loader.dataset, batch_size=memory_bs, shuffle=False,
            num_workers=train_loader.num_workers, pin_memory=True)
        lemniscate.memory = torch.zeros(len(train_loader.dataset), args.low_dim).cuda()
        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets, _, indexes) in enumerate(tqdm.tqdm(temploader)):
                batchSize = inputs.size(0)
                features = model(inputs)
                lemniscate.memory[batch_idx * batchSize:batch_idx * batchSize + batchSize, :] = features.data
        #train_loader.dataset.transform = transform_bak
        model.train()
        
    if args.recompute_memory and args.resume:
        save_checkpoint({
                'epoch': args.start_epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'lemniscate': lemniscate,
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, False, 'recomputed_{}'.format(args.resume))
    
    cudnn.benchmark = True

    if args.evaluate:
        kNN(model, lemniscate, train_loader, val_loader, args.K, args.nce_t)
        return

    begin_train_time = datetime.datetime.now()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        if epoch % 1 == 0:
            # evaluate on validation set
            #prec1 = NN(epoch, model, lemniscate, train_loader, train_loader) # was evaluating on train
            prec1 = kNN(model, lemniscate, train_loader, val_loader, args.K, args.nce_t)
            # prec1 really should be renamed to prec5 as kNN now returns top5 score, but
            # it won't be backward's compatible as earlier models were saved with "best_prec1"

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'lemniscate': lemniscate,
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)

        # train for one epoch
        train(train_loader, model, lemniscate, criterion, optimizer, epoch)

    # print elapsed time
    end_train_time = datetime.datetime.now()
    d = end_train_time - begin_train_time
    print("Trained for %d epochs. Elapsed time: %s days, %.2dh: %.2dm: %.2ds" %
          (len(range(args.start_epoch, args.epochs)),
           d.days, d.seconds // 3600, (d.seconds // 60) % 60, d.seconds % 60))

    # evaluate KNN after last epoch
    kNN(model, lemniscate, train_loader, val_loader, args.K, args.nce_t)


def train(train_loader, model, lemniscate, criterion, optimizer, epoch):
    #pkl_file = open('idx_counts_ratios.pkl', 'rb')
    #idx_counts_ratios = pickle.load(pkl_file)
    #pkl_file.close()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    
    #all_inputs = None

    end = time.time()
    optimizer.zero_grad()
    for i, (input, target, _, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        index = index.cuda()#(async=True)
        
        '''
        if all_inputs is None:
            all_inputs = np.array(input)
        else:
            all_inputs = np.concatenate([all_inputs, np.array(input)], axis=0)
        '''
        
        # compute output
        feature = model(input)
        output = lemniscate(feature, index)
        loss = criterion(output, index) / args.iter_size

        '''
        lr = args.lr * (0.5 ** (epoch // 2))
        
        targs = np.array(target.cpu().data)
        final_ratio = 0
        for targ in targs:
            final_ratio += 1./idx_counts_ratios[targ]
            
        final_ratio = final_ratio / len(targs)
        
        #print('targs: {}, final_ratio: {}'.format(targs, final_ratio))
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * final_ratio
        '''
            
        #Backprop Apex optimizer loss
        optimizer.backward(loss)

        # measure accuracy and record loss
        losses.update(loss.item() * args.iter_size, input.size(0))

        if (i+1) % args.iter_size == 0:
            # compute gradient and do SGD step
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss:.4f}\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=loss.item()))
    
    #np.save('all_train_inputs.npy', all_inputs) 


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    epoch = state['epoch']-1
    
    filename = 'batch{}_temp{}_epoch{}_{}'.format(args.batch_size, args.nce_t, epoch, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'batch{}_temp{}_epoch{}_model_best.pth.tar'.format(args.batch_size, args.nce_t, epoch))

def adjust_learning_rate(optimizer, epoch):
    """Decays the learning rate according to the default schedule, or more aggressively if fine-tuning"""

    if args.fine_tune:
        lr = args.lr * (0.5 ** (epoch // 2))
    else:
        lr = args.lr
        if epoch < 120:
            lr = args.lr
        elif epoch >= 120 and epoch < 160:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
        #lr = args.lr * (0.1 ** (epoch // 100))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
