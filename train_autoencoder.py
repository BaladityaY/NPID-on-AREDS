import argparse
import os
import sys
import shutil
import time
import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
import torchvision.models as models

import datasets
import models
import datetime

from models.autoencoder import Autoencoder
from lib.utils import AverageMeter

from collections import Counter
from apex.fp16_utils import FP16_Optimizer
from tqdm import tqdm

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--iter_size', default=1, type=int, help='caffe style iter size')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--low-dim', default=128, type=int, metavar='D', help='feature dimension')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')

parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--balanced_sampling', action='store_true', help='by default set false, oversamples less populated classes')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--static_loss', default=25, type=float, help='set static loss for apex optimizer')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--output-directory', default='./', type=str, help='directory for saved models')
parser.add_argument('--recompute-memory', default=False, action='store_true', help='recompute memory for evaluation stage')
parser.add_argument('--no-projection', default=False, action='store_true', help='test only')
parser.add_argument('--test-frequency', default=5, type=int, help='the frequency in epochs for how often to evaluate during train')

parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='Use pre-trained resnet model as encoder')
parser.add_argument('--pretrained-path', default='./lemniscate_resnet18.pth.tar', type=str, help='Directory for pretrained lemniscate model to pull encoder from')
parser.add_argument('--image-directory', default='./', type=str, help='Directory to write the images generated during validation')
parser.add_argument('--skip-conn', default=False, action='store_true', help='Use skip connections')


best_mse = 0


def main():
    global args, best_mse, image_dir
    best_mse = 9999999
    args = parser.parse_args()
    image_dir = args.image_directory

    args.distributed = args.world_size > 1
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
    print("=> creating autoencoder with model '{}'".format(args.arch))
    model = Autoencoder(arch=args.arch, low_dim=args.low_dim, skip_conn=args.skip_conn)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.234, 0.191, 0.159],  # xView stats
                                     std=[0.173, 0.143, 0.127])
    print("=> creating datasets")

    train_dataset = datasets.ImageFolderInstance(
        traindir,
        transforms.Compose([
            # transforms.RandomResizedCrop(224, scale=(0.2,1.)),
            # transforms.RandomGrayscale(p=0.2),
            transforms.Resize((224, 224)),
            transforms.ColorJitter(0.05, 0.05, 0.05, .05),  # transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolderInstance(
        valdir,
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = None

    if args.balanced_sampling:
        # Here's where we compute the weights for WeightedRandomSampler
        if args.concat_on_train:
            class_labels = [sample[1] for sample in train_dataset]
            class_counts = Counter(class_labels)
            class_reciprocals = {k: float(len(train_dataset)) / float(v) for k, v in class_counts.items()}
            reciprocal_weights = [class_reciprocals[label] for label in class_labels]
        else:
            class_counts = {v: 0 for v in train_dataset.class_to_idx.values()}
            for path, ndx in train_dataset.samples:
                class_counts[ndx] += 1
            total = float(np.sum([v for v in class_counts.values()]))
            class_probs = [class_counts[ndx] / total for ndx in range(len(class_counts))]
            # make a list of class probabilities corresponding to the entries in train_dataset.samples
            reciprocal_weights = [1. / class_probs[idx] for i, (_, idx) in enumerate(train_dataset.samples)]
        # weights are the reciprocal of the above
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(reciprocal_weights, len(train_dataset), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # Dont shuffle validation (for consistent image generation)- this has no effect on mse
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Model must be wrapped in DataParallel AFTER apex initialization and BEFORE resuming from checkpoint
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    if args.pretrained:
        print("=> using pre-trained encoder model '{}' from {}".format(args.arch, args.pretrained_path))
        pretrained = model.module.encoder
        if os.path.isfile(args.pretrained_path):
            print("=> loading checkpoint '{}'".format(args.pretrained_path))
            checkpoint = torch.load(args.pretrained_path)
            # The obvious solution here is to wrap pretrained in nn.DataParallel (DP)
            # That'd make the encoder itself wrapped in DP, which runs into
            # Issues when you try to wrap the whole autoencoder in DP

            # Solution: Change state dict keys to remove DP artifacts
            new_state_dict = {}
            for k in checkpoint['state_dict'].keys():
                new_state_dict[k.strip('module').lstrip('.')] = checkpoint['state_dict'][k]
            pretrained.load_state_dict(new_state_dict)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pretrained_path, checkpoint['epoch']))
            print('=> Reverting epoch count to 0 (to be trained for 200)')
            args.start_epoch = 0
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit(0)
        model.module.encoder = pretrained

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mse = checkpoint['best_mse']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.static_loss)
    cudnn.benchmark = True

    if args.evaluate:
        evaluate(val_loader, model, criterion, epoch=-1, write_img=True)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        if epoch % args.test_frequency == 0 and (epoch > 0 or args.test_frequency == 1):
            curr_mse = evaluate(val_loader, model, criterion, epoch, write_img=True)

            is_best = curr_mse < best_mse
            best_mse = min(curr_mse, best_mse)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_mse': best_mse,
                'optimizer': optimizer.state_dict(),
                'encoder_state_dict': model.module.encoder.state_dict(),
            }, is_best, filename=os.path.join(args.output_directory, 'checkpoint.pth.tar'))
    # evaluate after last epoch
    evaluate(train_loader, model, criterion, epoch=-1)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()

    optimizer.zero_grad()

    for i, (input, target, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        # compute output
        pred = model(input)

        loss = criterion(pred, input) / args.iter_size
        optimizer.backward(loss)

        if (i+1) % args.iter_size == 0:
            # compute gradient and do step
            optimizer.step()
            optimizer.zero_grad()

        # measure accuracy and record loss
        losses.update(loss.item() * args.iter_size, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('{0} => Epoch: [{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'AutoEncoder Loss {auto_loss:.4f}\t'.format(datetime.datetime.now(),
                                             epoch, i, len(train_loader), batch_time=batch_time,
                                             data_time=data_time, auto_loss=loss.item()))


def evaluate(val_loader, model, criterion, epoch=0, write_img=False):
    model.eval()
    bar = tqdm(iter(val_loader))
    mse_meter = AverageMeter()
    min_mse = 99999999
    best_reconstruct, best_input = None, None
    with torch.no_grad():
        for i, (input, _, _) in enumerate(bar):
            input = input.cuda(non_blocking=True)

            pred = model(input)
            loss = criterion(pred, input)
            # Store the items
            if loss.item() < min_mse and i < len(iter(val_loader)):
                best_reconstruct = pred
                best_input = input
                min_mse = loss.item()

            mse_meter.update(loss.item(), input.size(0))

    if write_img:
        # Write the best reconstruction and best input to files
        save_image(best_reconstruct, os.path.join(image_dir, 'reconstructed_epoch_{}.png'.format(epoch)))
        save_image(best_input, os.path.join(image_dir, 'input_epoch_{}.png'.format(epoch)))
    print('Evaluation complete with avg MSE: {}'.format(mse_meter.avg))
    return mse_meter.avg


def save_checkpoint(state, is_best, filename='./checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        head, tail = os.path.split(filename)
        shutil.copyfile(filename, os.path.join(head, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def make_sampler_for_balanced_classes(images, nclasses, replacement_method):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))

    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])

    weights = [0] * len(images)
    for idx, val in enumerate(images):
        weights[idx] = weight_per_class[val[1]]
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights), replacement=replacement_method)

    return sampler


if __name__ == '__main__':
    main()

