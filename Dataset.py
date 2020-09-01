import torch
from torch.utils.data import Dataset
from torchvision import models, transforms
import os
import random
import pandas as pd
import numpy as np
from collections import namedtuple
from skimage import io, transform
from PIL import Image
import h5py

import config
from utils import name_list

class SatUAVDataset(Dataset):
    '''
    Raw images + augmented pairs.
    '''

    def __init__(self, csv_meta, csv_file, root_dir=config.DATA_DIR, transform=None):
        self.meta = pd.read_csv(os.path.join(root_dir, csv_meta))
        self.path_list = list(self.meta.itertuples(index=False, name=None))
        self.file_frame = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform
        self.raw_len = len(self.file_frame) # number of raw images as descried in csv_file

    def __len__(self):
        return self.raw_len * len(self.path_list)

    @staticmethod
    def image_name(raw_name, aug_trick):
        assert len(raw_name) == 10, "raw_name is :"+raw_name+", whose len is not 10."
        if aug_trick.lower() == 'raw':
            return raw_name
        l = raw_name.split(".")
        l[0] += "_"+aug_trick[0].lower()
        return ".".join(l)

    def __getitem__(self, idx):

        # construct A related information
        A = {
            'aug_trick': self.path_list[idx//self.raw_len][0],
            'dir': os.path.join(self.root_dir, self.path_list[idx//self.raw_len][1]),
            'idx': idx % self.raw_len,
        }
        A['raw_name'] = self.file_frame.iloc[A['idx'],0]
        A['path'] = os.path.join(A['dir'], self.image_name(A['raw_name'], A['aug_trick']))

        # pick random augmentation trick for B, if unpaired, pick a shift for B
        rand_trick_idx = np.random.randint(len(self.path_list))
        label = [0] # paired : 0, unpaired : 1
        if random.choice([True, False]): # if unpaired
            shift = np.random.randint(low=1, high=self.raw_len)
            label[0] = 1
        else:
            shift = 0

        # construct B related information
        B = {
            'aug_trick': self.path_list[rand_trick_idx][0],
            'dir': os.path.join(self.root_dir, self.path_list[rand_trick_idx][1]),
            'idx': (idx+shift) % self.raw_len,
        }
        B['raw_name'] = self.file_frame.iloc[B['idx'], 1]
        B['path'] = os.path.join(B['dir'], self.image_name(B['raw_name'], B['aug_trick']))

        A_img = Image.open(A['path']).convert('RGB')
        B_img = Image.open(B['path']).convert('RGB')
        if self.transform:
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)
        sample = {'A': A_img, 'B': B_img, 'label': torch.FloatTensor(label)}

        return sample

class SatUAVH5Dataset(Dataset):

    def __init__(self, csv_file, feature_file):
        self.file_frame = pd.read_csv(csv_file)
        self.h5file = h5py.File(feature_file, 'r', swmr=True)
        self.len = len(self.file_frame)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        A_idx = int(self.file_frame.iloc[idx, 0].split('_')[0])
        label = [1]
        shift_idx, shift = idx, 0
        if random.choice([True, False]):
            shift = np.random.randint(low=1,high=self.len)
            shift_idx = (idx+shift) % self.len
            label[0] = 0
        B_idx = int(self.file_frame.iloc[shift_idx, 1].split('_')[0])

        A_tensor = torch.from_numpy(self.h5file['f_a'][A_idx-1:A_idx]).float()
        B_tensor = torch.from_numpy(self.h5file['f_b'][B_idx-1:B_idx]).float()

        sample = {'A': A_tensor, 'B': B_tensor, 'label': torch.FloatTensor(label)}
        #print(A_tensor.shape, B_tensor.shape, sample['label'].shape, idx, shift_idx, shift, A_idx, B_idx)

        return sample

# --------------------------- Deprecated --------------------------- #
class SatAerPairDataset(Dataset):

    def __init__(self, csv_file, root_dir=config.FULL_DATA, transform=None):
        quit("This SatAerPairDataset is deprecated. Please use SatAerDataset.")
        self.file_frame = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform
        self.len = len(self.file_frame)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        A_name = os.path.join(self.root_dir, self.file_frame.iloc[idx,0])
        A_img = Image.open(A_name)
        # label = [1,0]
        label = [1]
        B_idx = idx
        # if np.random.randint(2): # we want 50% data are negtive samples
        if random.choice([True, False]):
            shift = np.random.randint(low=1,high=self.len)
            B_idx = (idx+shift) % self.len
            # label = [0,1]
            label[0] = 0
        B_name = os.path.join(self.root_dir, self.file_frame.iloc[B_idx,1])
        B_img = Image.open(B_name)
        if self.transform:
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)
        sample = {'A': A_img, 'B': B_img, 'label': torch.FloatTensor(label)}

        return sample

class SatAerSiameseDataset(Dataset):

    def __init__(self, csv_file, root_dir=config.FULL_RESIZED, transform=None):
        quit("This SatAerSiameseDataset is deprecated. Please use SatAerDataset.")
        self.file_frame = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform
        self.len = len(self.file_frame)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        A_name = os.path.join(self.root_dir, self.file_frame.iloc[idx,0])
        A_img = Image.open(A_name).convert('RGB')
        label = [0] # paird : 0, unpaired : 1
        B_idx = idx
        if random.choice([True, False]): # if true, choose unpaird images
            shift = np.random.randint(low=1,high=self.len)
            B_idx = (idx+shift) % self.len
            label[0] = 1
        B_name = os.path.join(self.root_dir, self.file_frame.iloc[B_idx,1])
        B_img = Image.open(B_name).convert('RGB')
        if self.transform:
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)
        sample = {'A': A_img, 'B': B_img, 'label': torch.FloatTensor(label)}

        return sample

