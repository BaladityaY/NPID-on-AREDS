import numpy as np
#import cv2
import h5py
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from random import shuffle
import os
import random
import scipy
import scipy.misc
import pickle
import h5py
import skimage

#from PIL import Image

from skimage.filters import gaussian as gaussian_filter
#from skimage.filters import gaussian_filter
#from skimage import filter

def get_device(device_id = 0):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(device_id)
        torch.cuda.device(device_id)
        return device
    else:
        device = torch.device("cpu")
        device_name = "cpu"
        return device

class Dataset(data.Dataset):
    
    def sort_folder_ft(self, s):
        '''
        Returns the last two entries, file name and last folder name, as key to sort
        '''
        return s.split('/')[-2]+'/'+s.split('/')[-1]
        
    def sort_filelist(self, data_folder_dir):
        
        file_list = []
        for path, subdirs, files in os.walk(data_folder_dir,followlinks=True):
            for file_name in files:
                if file_name.endswith('h5'):
                    filename = os.path.join(path,file_name)
                    
                    file_list.append(filename)
                            
        return sorted(file_list,key=self.sort_folder_ft)
        
        
    
    
    def __init__(self, data_dir, transform=None, condition=1, fut_cond=0):
        img_x = 2300; img_y = 3400
        img_x = img_x/10; img_y = img_y/10
        
        self.transform = transform
        self.condition = condition
        self.future_condition = fut_cond
        
        print('condition : {}'.format(self.condition))
        print('future condition : {}'.format(self.future_condition))
        
        self.data_len = 0
        
        self.fname = data_dir+'amd_9step.hdf5'
        
        self.aux_fname = data_dir+'aux_info.hdf5'
        self.aux_keys = ['AMDSEV', 'DRARWI', 'DRSZWI', 'GEOACT', 'GEOAWI', 'NUCADJ', 'NUCSCL', 'PCTCOA', 'PCTCOL', 'PCTPSC']
        
        with h5py.File(self.fname, 'r') as f:
            self.data_len = f['scores'].shape[0]
            
        
    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        train_x = 224; train_y = 224
        
        with h5py.File(self.fname, 'r') as f:
            imgs = f['imgs']
            scores = f['scores']
            years = f['years']
            
            #all_min = -183.48595
            
            '''
            cs = f['cs']
            ns = f['ns']
            psc = f['psc']
            '''
            
            img = np.array(imgs[index:index+1,:,:,:])[0]
            #img = np.array(Image.fromarray(img).resize((train_x, train_y)))
            img = scipy.misc.imresize(img, (train_x, train_y))
            #print('img shape: {}'.format(img.shape))
            
            real_base_img = img.copy()
            
            if not (self.transform is None):
                img = self.transform(img)
                
                img = np.array(img)
                img = np.transpose(img, (1, 2, 0))

            #img = np.clip(img, -1, 1)
            og_img = img.copy() #skimage.color.rgb2lab(img)[:,:,0]

            sigma = 9
            
            all_min_old = -40
            all_max_old = 40 #4.56
            
            all_min_new = -2
            all_max_new = 2
            
            img = og_img - ((all_max_old - all_min_old)*gaussian_filter((og_img - all_min_old)/(all_max_old - all_min_old), sigma=sigma, multichannel=True) + all_min_old)
            
            mid_min = all_min_old
            mid_max = all_max_old
            img = ((img - mid_min)/(mid_max - mid_min))*(all_max_new - all_min_new) + all_min_new
            
            #img = og_img - 255.*gaussian_filter(og_img/255., sigma=sigma, multichannel=True)
            img = img.transpose((2, 0, 1))
            #img = (img/255.) - .5
            #img = img - all_min # keep positive for np.uint8 conversion
            
            #print('dataset img shape: {}'.format(img.shape))
            
            '''
            imgs = []
            for i in range(3):
                #sigma = 3
                
                img = og_img - 255.*gaussian_filter(og_img/255., sigma=sigma, multichannel=True)
                #img = og_img - 255.*filter.gaussian_filter(og_img/255., sigma=sigma, multichannel=True)
                
                img = (img/255.) - .5
                
                imgs.append(img)
                
            imgs = np.array(imgs)#.astype(float32)    
            img = imgs 
            '''
            
            score = np.array(scores[index:index+1])[0]
            
            ##scores = self.future_labels() #f['scores']
            ##score = np.array(scores[index:index+1])[0] 
            
            if self.condition == 1:
                score = np.max((np.min((score - 1, 11)), 0)) #np.max((0, score - 1))
                
            elif self.condition == 2:   
                score = np.max((np.min(((score - 1)/3, 3)), 0)).astype(int) #np.max((0, score - 1))
                
            elif self.condition == 3:   
                #'''
                score = np.max((0, score - 1)).astype(int) # now from 0-11

                if score < 6: # changed from 9 to 6
                    while score > 0:
                        score = score - 1

                if score >= 6: # changed from 9 to 6
                    while score > 1:
                        score = score - 1
                #'''
                
            elif self.condition == 4:   
                #'''
                score = np.max((0, score - 1)).astype(int) # now from 0-11

                if score < 9: # changed from 9 to 6
                    while score > 0:
                        score = score - 1

                if score >= 9: # changed from 9 to 6
                    while score > 1:
                        score = score - 1
                #'''
                
            elif self.condition == 5:   ## like Yan et al
                #'''
                score = np.max((0, score - 1)).astype(int) # now from 0-11

                if score > 1 and score < 8: # changed from 1-7 -> 1
                    while score > 1:
                        score = score - 1

                if score >= 8: # changed from 8-11 -> 2
                    while score > 2:
                        score = score - 1
                #'''
            
            year = np.array(years[index:index+1])[0]

            '''
            cs = np.array(cs[index:index+1])[0]
            cs = cs/5
            cs = np.max((np.min((cs, 20)), 0))
            
            ns = np.array(ns[index:index+1])[0]
            #ns = ns/1
            ns = np.max((np.min((ns, 6)), 0))
            
            psc = np.array(psc[index:index+1])[0]
            psc = ns/3
            psc = np.max((np.min((ns, 30)), 0))
            '''
            
            #scores = self.future_labels() #f['scores']
            #score = np.array(scores[index:index+1])[0] # from future conditon, to delete if not used
            #print(score)
            
            #img = img.astype(np.float32) # (np.uint8)
            score = score.astype(int)
            year = year.astype(int)
            '''
            cs = cs.astype(int)
            ns = ns.astype(int)
            psc = psc.astype(int)
            '''
            
        return img, score, year, index #, cs, ns, psc, index
    
    @property
    def train_labels(self):
        with h5py.File(self.fname, 'r') as f:
            scores = np.array(f['scores'])
            
            
        ##scores = self.future_labels()
        
        if self.condition == 1:
            scores = scores - 1
            scores[scores<0] = 0
            scores[scores>11] = 11
            
        elif self.condition == 2:
            scores = scores - 1
            scores = scores / 3
            scores[scores<0] = 0
            scores[scores>3] = 3
            scores = scores.astype(int)
        
        elif self.condition == 3:
            scores = scores - 1

            scores[scores < 6] = 0 # changed from 9 to 6
            scores[scores >= 6] = 1
            scores = scores.astype(int)
        
        elif self.condition == 4:
            scores = scores - 1

            scores[scores < 9] = 0 # changed from 9 to 6
            scores[scores >= 9] = 1
            scores = scores.astype(int)
            
        elif self.condition == 5: ## like Yan et al
            scores = scores - 1

            scores[scores<0] = 0
            scores[(scores > 1) & (scores < 8)] = 1 # changed from 1-7 -> 1
            scores[scores >= 8] = 2 # changed from 8-11 -> 2
            scores = scores.astype(int)
        
        #scores = self.future_labels()
        #scores = scores.astype(int)
        
        return scores
    
    def future_condition(self):
        return self.future_condition
    
    def train_years(self):
        with h5py.File(self.fname, 'r') as f:
            years = np.array(f['years'])
            
        return years
    
    def dir_ids(self):
        with h5py.File(self.fname, 'r') as f:
            dir_ids = np.array(f['dir_ids'])
            
        return dir_ids
    
    def condition(self):
        return self.condition
    
    def future_labels(self):
        fut_cond = self.future_condition
        
        if fut_cond == 0:
            name = 'referrable_any'
            
        if fut_cond == 1:
            name = 'referrable_5yr'
            
        if fut_cond == 2:
            name = 'advanced_any'
            
        if fut_cond == 3:
            name = 'advanced_5yr'
            
        if fut_cond == 4:
            name = 'ga_any'
            
        if fut_cond == 5:
            name = 'ga_5yr'
            
        if fut_cond == 6:
            name = 'mnv_any'
            
        if fut_cond == 7:
            name = 'mnv_5yr'
            
        if fut_cond == 8:
            name = 'ma_any'
            
        if fut_cond == 9:
            name = 'ma_5yr'
            
        #with h5py.File(self.fname, 'r') as f:
        #    fut_labs = np.array(f[name])
            
            
        
        with h5py.File(self.aux_fname, 'r') as f:
            fut_labs = np.array(f[self.aux_keys[fut_cond]])
            fut_labs = fut_labs.astype(int)
            
            fut_labs = fut_labs + 1
            fut_labs[fut_labs<0] = 0
            
            print('EVALUATING {} NOW'.format(self.aux_keys[self.future_condition]))
            
        return fut_labs


if __name__ == '__main__':
    
    dset = Dataset('/home/bala/areds/AREDS/net_training/data/val/')
    
    train_loader = torch.utils.data.DataLoader(dset, batch_size=4, shuffle=False, num_workers=0)
    
    for i, (imgs, scores, years, indices) in enumerate(train_loader):
        print('imgs shape: {} \nscores: {} \nyears:{} \nindices: {} \n\n'.format(imgs.shape, scores, years, indices))
    
