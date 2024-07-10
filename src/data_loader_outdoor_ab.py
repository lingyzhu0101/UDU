#-*-coding:utf-8-*-

from pathlib import Path
from itertools import chain
import os

import glob
import random
from munch import Munch
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms

def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext)) for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


class ReferenceDataset(data.Dataset):
    def __init__(self, root, sample_frames=5, transform=None):
        super(ReferenceDataset, self).__init__()
        self.sample_frames = sample_frames    
        self.transform = transform
        self.task_videos_exp = []
        self.num_frames_exp = []
        self.task_videos_SDSD = []
        self.num_frames_SDSD = []
        
        # read raw
        videos_SDSD = [line.rstrip('\n') for line in open(root + './SDSD_CUHK/train_list_outdoor.txt')] # train_list_indoor train_list_outdoor
        for video_SDSD in videos_SDSD:
            self.task_videos_SDSD.append([os.path.join(root, "SDSD_CUHK/videoSDSD_low_resize", video_SDSD)])
            input_dir = os.path.join(root, "./SDSD_CUHK/videoSDSD_low_resize", video_SDSD)
            frame_list = glob.glob(os.path.join(input_dir, '*.*'))
            self.num_frames_SDSD.append(len(frame_list))
        print("[%s] Total %d SDSD training videos (%d frames)" %(self.__class__.__name__, len(self.task_videos_SDSD), sum(self.num_frames_SDSD)))
        
        # read exp one video
        self.task_videos_exp.append([os.path.join(root, "fiveK/")])
        while (len(self.task_videos_exp) < len(self.task_videos_SDSD)):
            self.task_videos_exp += self.task_videos_exp.copy() 
        self.task_videos_exp = self.task_videos_exp[0:len(self.task_videos_SDSD)] 
        for i in range(len(self.task_videos_exp)):
            frame_list = glob.glob( os.path.join(self.task_videos_exp[i][0], '*.*'))
            self.num_frames_exp.append(len(frame_list))
        print("[%s] Total %d exp training videos (%d frames)" %(self.__class__.__name__, len(self.task_videos_exp), sum(self.num_frames_exp)))

    def __getitem__(self, index):
    
        ## random select starting frame index t between [0, N - #sample_frames]
        N_SDSD = self.num_frames_SDSD[index]
        N_exp = self.num_frames_exp[index]
        T_SDSD = random.randint(0, N_SDSD - self.sample_frames)
        T_exp = random.randint(0, N_exp - self.sample_frames)
    
        
        path_SDSD = self.task_videos_SDSD[index][0]
        path_exp = self.task_videos_exp[index][0]
         
        frame_low_SDSD = []      
        filename_low_SDSD = [] 
        frame_normal_exp = [] 
        
        for t in range(T_SDSD, T_SDSD + self.sample_frames):
            img_SDSD = Image.open(os.path.join(path_SDSD, "%05d.png" % t)).convert('RGB')
            frame_low_SDSD.append(self.transform(img_SDSD))
            
            # get video and frame name
            frame_path, _ = str(os.path.join(path_SDSD, "%05d.png" % t)).split('.', 1)
            video_name = frame_path.rsplit('/')[-2]
            frame_name = frame_path.rsplit('/')[-1]
            img_name = video_name + "_" + frame_name
            filename_low_SDSD.append(img_name)
            
        
        for t in range(T_exp, T_exp + self.sample_frames): # (512, 960) (512, 960, 3)
            frame_list = glob.glob(os.path.join( path_exp, '*.*'))
            img_exp =  Image.open(frame_list[t]).convert('RGB')
            frame_normal_exp.append(self.transform(img_exp))
            
        return frame_normal_exp, frame_low_SDSD, filename_low_SDSD
        
    def __len__(self):
        return len(self.task_videos_SDSD)



def get_train_loader(root, sample_frames=5, img_size=512, resize_size=256, batch_size=8, shuffle=True, num_workers=8, drop_last=True):

    transform = transforms.Compose([
        transforms.RandomCrop(img_size),
        transforms.Resize([resize_size, resize_size]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = ReferenceDataset(root, sample_frames, transform)

    return data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=drop_last)
                           
                           
                           
class ReferenceDataset_Val(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = self._make_dataset(root)
        self.transform = transform

    def _make_dataset(self, root):
        fnames, fnames2 = [], []
        
        videos_SDSD = [line.rstrip('\n') for line in open(root + './SDSD_CUHK/test_list_outdoor.txt')] # test_list_outdoor
        for video_SDSD in videos_SDSD:  
            class_dir_exp = os.path.join(root, "SDSD_CUHK/videoSDSD_normal_resize", video_SDSD, "00002.png")
            fnames.append(class_dir_exp)
              
            class_dir_low = os.path.join(root, "SDSD_CUHK/videoSDSD_low_resize", video_SDSD, "00002.png")
            fnames2.append(class_dir_low)
    
        return list(zip(fnames, fnames2))

    def __getitem__(self, index):    
        fname, fname2 = self.samples[index]
        # get video and frame name
        frame_path, _ = str(fname2).split('.', 1)
       
        video_name = frame_path.rsplit('/')[-2]
        frame_name = frame_path.rsplit('/')[-1]
        img_name = video_name + "_" + frame_name
       
        # read and transform
        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, img_name

    def __len__(self):
        return len(self.samples)
        

def get_val_loader(root, img_size=512, batch_size=8, shuffle=False, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize([288,512]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = ReferenceDataset_Val(root, transform)
    return data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


class InputFetcher:
    def __init__(self, loader):
        self.loader = loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _fetch_refs(self):
        try:
            x, y, name = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y, name = next(self.iter)
        return x, y, name

    def __next__(self):
        x, y, img_name = self._fetch_refs()
        #x, y = x.to(self.device), y.to(self.device)
        inputs = Munch(img_exp=x, img_raw=y, img_name=img_name)
        
        return inputs
        
        