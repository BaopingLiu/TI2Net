# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 15:15:37 2022

@author: Baoping Liu
"""
import torch
from torch import nn
import numpy as np
from config.config import Config
from torch.nn import DataParallel, init
from models import resnet_face18
import torch.nn.init as init
from torch import isnan

def load_arcface():
    opt = Config()
    model = resnet_face18(opt.use_se)
    model = DataParallel(model)
    model.load_state_dict(torch.load(opt.test_model_path))
    model.to(torch.device("cuda"))
    return model

class LandmarkDropout(nn.Module):
    """
    from LRNet: https://github.com/frederickszk/LRNet
    """
    def __init__(self, p: float = 0.5):
        super(LandmarkDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p

    def generate_mask(self, landmark, frame):
        position_p = torch.bernoulli(torch.Tensor([1 - self.p]*(landmark//2)))
        return position_p.repeat(1, frame, 2)

    def forward(self, x: torch.Tensor):
        if self.training:
            _, frame, landmark = x.size()
            landmark_mask = self.generate_mask(landmark, frame)
            scale = 1/(1-self.p)
            return x*landmark_mask.to(x.device)*scale
        else:
            return x
        

class TI2Net(nn.Module):#Identity Inconsistency Detection
    def __init__(self, feature_dim=1024, rnn_unit=244, dropout_rate=0.5, num_layers=1):
        super(TI2Net, self).__init__()
        self.dropout_landmark = LandmarkDropout(dropout_rate)
        self.feature_dim = feature_dim
        self.rnn = nn.GRU(input_size=self.feature_dim, 
                          hidden_size=rnn_unit, 
                          num_layers=num_layers,
                          batch_first=True, 
                          bidirectional=True)
        self.dropout_feature = nn.Dropout(0.7)
        self.linear_1 = nn.Linear(2*rnn_unit, 128)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(128, 2)
        self.output = nn.Softmax(dim=1)
        #self.output = nn.Sigmoid()
        self.initialize()
    
    def initialize(self):
        self.rnn.weight_ih_l0.data.normal_(0.0, 0.1)
        self.rnn.weight_hh_l0.data.normal_(0.0, 0.1)  
        self.rnn.bias_ih_l0.data.fill_(0.0)  
        self.rnn.bias_hh_l0.data.fill_(0.0)  
        init.xavier_uniform_(self.linear_1.weight)
        init.zeros_(self.linear_1.bias)
        init.xavier_uniform_(self.linear_2.weight)
        init.zeros_(self.linear_2.bias)
    
    
    def forward(self, x, diff_x=None):
        x = self.dropout_landmark(x)
        rnn_embeddings, _ = self.rnn(x)
        latent_feat = rnn_embeddings[:, -1, :]
        x = self.dropout_feature(latent_feat)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.output(x)
        return  latent_feat, x
    

