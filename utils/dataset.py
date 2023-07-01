# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 15:15:37 2022

@author: Baoping Liu
"""
import torch
from torch.utils import data
import numpy as np
from random import randint
import torchvision.transforms as transforms
from torch.nn.functional import normalize

def normalize_minmax(vec):
    min_val = torch.min(vec)
    max_val = torch.max(vec)
    res = (vec - min_val) / (max_val - min_val)
    return res

def Normalize(vec):
    mean = torch.mean(vec, dim=0)
    std = torch.std(vec, dim=0)
    normalized_features = (vec - mean) / std
    return normalized_features


class Pair_Dataset(data.Dataset):
    def __init__(self, real_id_set, fake_id_set, phase=None):
        self.phase  = phase
        self.seq_step = 2
        self.seq_len = 64
        self.real_video_ids = real_id_set
        self.fake_video_ids = fake_id_set
    
    def sample_seq(self, video_ids):
        upper_bound = video_ids.shape[0]-(self.seq_len+1)*self.seq_step
        start = np.random.randint(0, upper_bound)
        idx_list = [start+i*self.seq_step for i in range(self.seq_len)]
        seq = video_ids[idx_list]
        seq = self.diff_ids_sequence(seq)
        return torch.Tensor(seq)
    
    def generate_train_seq(self, anc_idx, pos_idx, neg_idx):
        anchor_video_ids = np.array(self.real_video_ids[anc_idx])
        positive_video_ids = np.array(self.real_video_ids[pos_idx])
        negative_video_ids = np.array(self.fake_video_ids[neg_idx])
        anchor_seq = self.sample_seq(anchor_video_ids)
        positi_seq = self.sample_seq(positive_video_ids)
        negati_seq = self.sample_seq(negative_video_ids)
        return anchor_seq, positi_seq, negati_seq
    
    def diff_ids_sequence(self, video_ids):
        return np.diff(video_ids, axis=0)
    
    def generate_test_seq(self, real_index, fake_index):
        real_video_ids = np.array(self.real_video_ids[real_index])
        fake_video_ids = np.array(self.fake_video_ids[fake_index])
        real_seq = self.sample_seq(real_video_ids)
        fake_seq = self.sample_seq(fake_video_ids)
        return real_seq, fake_seq
  
    def __getitem__(self, index):
        if self.phase=="train":
            anchor = len(self.fake_video_ids) % len(self.real_video_ids)
            pos = randint(0, len(self.real_video_ids)-1)
            while pos==anchor:
                #make sure anchor and positive is not from the same video
                pos = randint(0, len(self.real_video_ids)-1) 
            neg = randint(0, len(self.fake_video_ids)-1)
            anchor_seq, positi_seq, negati_seq = self.generate_train_seq(anchor, pos, neg)
            return anchor_seq, positi_seq, negati_seq, torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([1]), anchor, pos, neg #source of video for calculating video-level metrics
        elif self.phase=="test":
            real_idx = len(self.fake_video_ids) % len(self.real_video_ids)
            fake_idx = index
            real_seq, fake_seq = self.generate_test_seq(real_idx, fake_idx)
            return real_seq, fake_seq, \
                    torch.Tensor([0]), torch.Tensor([1]), \
                    real_idx, fake_idx
        else:
            print("Unsupported phase param")
            
    def __len__(self):
        return len(self.fake_video_ids)
