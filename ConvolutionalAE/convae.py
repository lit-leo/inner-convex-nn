import os
from torch.utils.data import DataLoader
#from torch.utils.data import random_split
from utilsdata import random_split
from torch.autograd import Variable
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.datasets as dset
import torchvision
# from sklearn.metrics import accuracy_score
# import sklearn
import torch.utils.data as torch_data
import torch.nn as nn
# from sklearn.model_selection import train_test_split
# import pandas as pd
import numpy as np
# from sklearn.datasets import load_digits
import torch
# import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from poslayers.poslayers import Dense, PosDense, PosConv2d
# from vae.vanila_vae import *

IMAGE_SIZE = 64 * 64
IMAGE_WIDTH = IMAGE_HEIGHT = 64
INPUT_CHANNELS = 3
DATA_PATH = r'./celeba_40k/'


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImageFolder(DATA_PATH, transform=transform)

# for dataset reduction
# dataset, _ = random_split(
#     dataset, (int(len(dataset) * 0.01), len(dataset) - int(len(dataset) * 0.01)))

train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, (train_size, val_size))


train_loader = DataLoader(dataset=train_data, batch_size=128,
                          shuffle=True, num_workers=4, drop_last=True)
val_loader = DataLoader(dataset=val_data, batch_size=128,
                        shuffle=True, num_workers=4, drop_last=True)


class AutoEncoder(nn.Module):
    
    def __init__(self, code_size):
        super().__init__()
        self.code_size = code_size
        
        # Encoder specification
        self.enc_cnn_1 = nn.Conv2d(INPUT_CHANNELS, 5, kernel_size=5)
        self.enc_cnn_2 = nn.Conv2d(5, 10, kernel_size=5)
        self.enc_linear_1 = Dense(10 * 13 * 13, 800)
        self.enc_linear_2 = Dense(800, self.code_size)
        
        # Decoder specification
        self.dec_linear_1 = Dense(self.code_size, 4000)
        self.dec_linear_2 = Dense(4000, IMAGE_SIZE * INPUT_CHANNELS)
        
    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return out, code
    
    def encode(self, images):
        code = self.enc_cnn_1(images)
        code = F.selu(F.max_pool2d(code, 2))
        
        code = self.enc_cnn_2(code)
        code = F.selu(F.max_pool2d(code, 2))
        
        code = code.view([images.size(0), -1])
        code = F.selu(self.enc_linear_1(code))
        code = self.enc_linear_2(code)
        return code
    
    def decode(self, code):
        out = F.selu(self.dec_linear_1(code))
        out = F.selu(self.dec_linear_2(out))
        out = out.view([code.size(0), INPUT_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT])
        return out

#    def l1reg(self, device):
 #       return self.enc_linear_1.l1reg(device) + 
  #             self.enc_linear_2.l1reg(device) + 
   #            self.dec_linear_1.l1reg(device) +
    #           self.dec_linear_2.l1reg(device)


# primitive one
class PosAutoEncoder(nn.Module):

    def __init__(self, code_size):
        super().__init__()
        self.code_size = code_size

        # Encoder specification
        self.enc_cnn_1 = PosConv2d(INPUT_CHANNELS, 5, kernel_size=5)
        self.enc_cnn_2 = PosConv2d(5, 10, kernel_size=5)
        self.enc_linear_1 = PosDense(10 * 13 * 13, 800)
        self.enc_linear_2 = PosDense(800, self.code_size)

        # Decoder specification
        self.dec_linear_1 = PosDense(self.code_size, 4000)
        self.dec_linear_2 = Dense(4000, IMAGE_SIZE * INPUT_CHANNELS)

    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return out, code

    def encode(self, images):
        code = self.enc_cnn_1(images)
        code = F.selu(F.max_pool2d(code, 2))

        code = self.enc_cnn_2(code)
        code = F.selu(F.max_pool2d(code, 2))

        code = code.view([images.size(0), -1])
        code = F.selu(self.enc_linear_1(code))
        code = self.enc_linear_2(code)
        return code

    def decode(self, code):
        out = F.selu(self.dec_linear_1(code))
        out = F.selu(self.dec_linear_2(out))
        out = out.view([code.size(0), INPUT_CHANNELS,
                        IMAGE_WIDTH, IMAGE_HEIGHT])
        return out

    def positivate(self):
        self.enc_cnn_1.positivate()
        self.enc_cnn_2.positivate()
        self.enc_linear_1.positivate()

    def l1reg(self, device):
        return self.enc_linear_1.l1reg(device) + self.enc_linear_2.l1reg(device) + self.dec_linear_1.l1reg(device) + self.dec_linear_2.l1reg(device)


# advanced one
class PosAE_advanced(nn.Module):
    
    def __init__(self, code_size):
        super().__init__()
        self.code_size = code_size
        
        # Encoder specification
        self.enc_cnn_1 = PosConv2d(INPUT_CHANNELS, 5, kernel_size=5) # -> (-4, -4) / 2
        self.enc_cnn_2 = PosConv2d(5, 10, kernel_size=5) # -> (-4, -4) / 2
        self.imdim = ((IMAGE_WIDTH - 4) // 2 - 4) // 2
        self.enc_linear_1 = PosDense(10 * self.imdim * self.imdim, 800)
        self.enc_linear_2 = Dense(800, self.code_size)
        
        # Decoder specification
        self.dec_linear_1 = nn.Linear(self.code_size, 800)
        self.dec_linear_2 = nn.Linear(800, 10 * self.imdim * self.imdim)
        
        self.dec_up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.dec_cnn_1 = nn.ConvTranspose2d(10, 5, kernel_size=5)
        
        self.dec_up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.dec_cnn_2 = nn.ConvTranspose2d(5, INPUT_CHANNELS, kernel_size=5)
        
    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return out, code
    
    def encode(self, images):
        code = self.enc_cnn_1(images)
        code = F.relu(F.max_pool2d(code, 2))
        
        code = self.enc_cnn_2(code)
        code = F.relu(F.max_pool2d(code, 2))
        
        code = code.view([images.size(0), -1])
        code = F.relu(self.enc_linear_1(code))
        code = self.enc_linear_2(code)
        return code
    
    def decode(self, code):
        out = F.relu(self.dec_linear_1(code))
        out = F.relu(self.dec_linear_2(out))
        out = out.view([-1, 10, self.imdim, self.imdim])
        out = self.dec_up1(out)
        out = F.relu(self.dec_cnn_1(out))
        out = self.dec_up2(out)
        out = F.relu(self.dec_cnn_2(out))
        out = out.view([-1, INPUT_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT])
        return out


    def positivate(self):
        self.enc_cnn_1.positivate()
        self.enc_cnn_2.positivate()
        self.enc_linear_1.positivate()

# Hyperparameters
code_size = 500
lr = 0.004

device = torch.device('cuda:0')
net = PosAutoEncoder(code_size=code_size).to(device)
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(net.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 7, 0.5)

def train_ae(epochs, net, criterion, optimizer, train_loader,
               val_loader, ds_train, ds_val,
               scheduler=None, verbose=True, save_dir=None, l1alpha=0, model_file='model', positivate=False):
    net.to(device)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_loss = np.inf

    for epoch in tqdm(range(1, epochs+1)):
        net.train()
        loss = []
        train_mse = []
        for X, y in (train_loader):
            X = X.to(device)
            #y = y.to(device)
            nn_outputs, code = net(X)
            mse = criterion(nn_outputs, X)
            loss1 = mse + l1alpha * net.l1reg(device)
            loss1 = loss1.to(device)
            optimizer.zero_grad()
            loss1.backward()
            loss.append(loss1.item())
            train_mse.append(mse.item())
            optimizer.step()
            if positivate:
              net.positivate()
        net.eval()
        val_loss = []
        mse_loss = []
        for X, y in val_loader:
            X = X.to(device)
            #y = y.to(device)
            nn_outputs, code = net(X)
            mse = criterion(nn_outputs, X)
            val_loss1 = mse + l1alpha * net.l1reg(device) 
            val_loss.append(val_loss1.item())
            mse_loss.append(mse.item())
        if scheduler is not None:
            scheduler.step()
        freq = 1

        train_losses.append(np.mean(loss))
        val_losses.append(np.mean(val_loss))
        train_accs.append(np.mean(train_mse))
        val_accs.append(np.mean(mse_loss))

        if np.mean(val_loss) < best_val_loss:
          best_val_loss = np.mean(val_loss)
          torch.save(net.state_dict(), model_file)

        if verbose and epoch%freq==0:
          print('Epoch {}/{} || Loss: Train {:.4f} | Val {:.4f} | Val MSE {:.4f}'.format(epoch, epochs, np.mean(loss), np.mean(val_loss), np.mean(mse_loss)))
          
    np.save(f'val_loss_{model_file}', val_losses)
    np.save(f'train_loss_{model_file}', train_losses)
    np.save(f'val_acc_{model_file}', val_accs)
    np.save(f'train_acc_{model_file}', train_accs)

alpha = 2e-6
train_ae(30, net, criterion, optimizer, train_loader, val_loader, train_data, val_data, scheduler=scheduler, model_file='convae', l1alpha = 0, positivate=False)

