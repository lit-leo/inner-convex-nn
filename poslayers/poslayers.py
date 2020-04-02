import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.utils.data as torch_data
import sklearn
from sklearn.metrics import accuracy_score

class Dense(nn.Linear):
    def __init__(self, *args,**kwargs):
        super(PosDense, self).__init__(*args,**kwargs)
    
    def get_sparsity(self):
        data = self.weight.data
        return (data == 0).sum().item() / (data.shape[0] * data.shape[1])
    
    def l1reg(self):
        l1_reg = torch.tensor(0., requires_grad=True)
        for name, W in self.named_parameters():
            l1_reg = l1_reg + W.norm(1)
        return l1_reg

class PosDense(nn.Linear):
    def __init__(self, *args,**kwargs):
        super(PosDense, self).__init__(*args,**kwargs)
    
    def positivate(self):
        self.weight.data = F.relu(self.weight.data)
    
    def get_sparsity(self):
        data = self.weight.data
        return (data == 0).sum().item() / (data.shape[0] * data.shape[1])
    
    def l1reg(self):
        l1_reg = torch.tensor(0., requires_grad=True)
        for name, W in self.named_parameters():
            l1_reg = l1_reg + W.norm(1)
        return l1_reg

class PosConv2d(nn.Conv2d):
    def __init__(self, *args,**kwargs):
        super(PosConv2d, self).__init__(*args,**kwargs)
    
    def positivate(self):
        self.weight.data = F.relu(self.weight.data)