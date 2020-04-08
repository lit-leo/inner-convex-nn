import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import random
import torch
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from   torchvision import datasets, transforms
# from torch.utils.data import random_split
from utilsdata import random_split
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm


train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(65),
      transforms.CenterCrop(64),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

catsdogs_train = datasets.ImageFolder('Data/PetImages',  
                                    transform=train_transforms)                                       
catsdogs_val = datasets.ImageFolder('Data/Val', 
                                    transform=test_transforms)

train_loader = torch_data.DataLoader(catsdogs_train, batch_size=100, shuffle=True) 
val_loader = torch_data.DataLoader(catsdogs_val, batch_size=100, shuffle=False) 

def get_accuracy(net, val_dset):
    test_loader = torch_data.DataLoader(val_dset,batch_size = len(val_dset)) 
    net.eval()
    for X,y in val_loader:
        X = X.to(device)
        y = y.to(device)
        nn_outputs = np.argmax(net(X).detach().cpu(), axis = 1)
    return accuracy_score(nn_outputs,y.detach().cpu())


class Dense(nn.Linear):
    def __init__(self, *args,**kwargs):
        super(Dense, self).__init__(*args,**kwargs)
    
    def get_sparsity(self):
        data = self.weight.data
        return (data == 0).sum().item() / (data.shape[0] * data.shape[1])
    
    def l1reg(self, device='cpu'):
        l1_reg = torch.tensor(0., requires_grad=True).to(device)
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
    
    def l1reg(self, device):
        l1_reg = torch.tensor(0., requires_grad=True).to(device)
        for name, W in self.named_parameters():
            l1_reg = l1_reg + W.norm(1)
        return l1_reg

class PosConv2d(nn.Conv2d):
    def __init__(self, *args,**kwargs):
        super(PosConv2d, self).__init__(*args,**kwargs)
    
    def positivate(self):
        self.weight.data = F.relu(self.weight.data)


class PosAlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(PosAlexNet, self).__init__()
        self.conv = PosConv2d(32, 64, kernel_size=3)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(32), 
            self.conv,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64)
        )
        
        self.fc1 = PosDense(12544, 512)
        self.fc2 = Dense(512, num_classes)

        
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            self.fc1,
            nn.ReLU(inplace=True),
            self.fc2,
        )

    def forward(self, x):
        conv_features = self.features(x)
        flatten = conv_features.view(conv_features.size(0), -1)
        fc = self.fc_layers(flatten)
        return fc
    
    def l1reg(self, device):
        return self.fc1.l1reg(device) + self.fc2.l1reg(device)
    
    def get_sparsities(self):
        return {
                'fc1': self.fc1.get_sparsity(),
                'fc2': self.fc2.get_sparsity()
        }
    
    def positivate(self):
      self.conv.positivate()
      self.fc1.positivate()


class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(32), 
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64)
        )
        
        self.fc1 = Dense(12544, 512)
        self.fc2 = Dense(512, num_classes)

        
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            self.fc1,
            nn.ReLU(inplace=True),
            self.fc2,
        )

    def forward(self, x):
        conv_features = self.features(x)
        flatten = conv_features.view(conv_features.size(0), -1)
        fc = self.fc_layers(flatten)
        return fc
    
    def l1reg(self, device):
        return self.fc1.l1reg(device) + self.fc2.l1reg(device)
    
    def get_sparsities(self):
        return {
                'fc1': self.fc1.get_sparsity(),
                'fc2': self.fc2.get_sparsity()
        }


device = 'cuda'

lr = 0.001
net = PosAlexNet(num_classes=2)  
criterion = F.cross_entropy
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
scheduler = None #torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.7)


def train_anet(epochs, net, criterion, optimizer, train_loader,
               val_loader, ds_train, ds_val,
               scheduler=None, verbose=True, save_dir=None, l1alpha=0,
               model_file='model', positivate=False, path=''):
    net.to(device)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_loss = np.inf

    for epoch in tqdm(range(1, epochs+1)):
        net.train()
        loss = []
        for X, y in (train_loader):
            X = X.to(device)
            y = y.to(device)
            nn_outputs = net(X)

            loss1 = criterion(nn_outputs, y) + l1alpha * net.l1reg(device)
            loss1 = loss1.to(device)
            optimizer.zero_grad()
            loss1.backward()
            loss.append(loss1.item())
            optimizer.step()
            if positivate:
              net.positivate()
        net.eval()
        val_loss = []
        for X, y in (val_loader):
            X = X.to(device)
            y = y.to(device)
            nn_outputs = net(X)
            val_loss1 = criterion(nn_outputs,y)
            val_loss.append(val_loss1.item())
         
        if scheduler is not None:
            scheduler.step()
        freq = 1

        train_losses.append(np.mean(loss))
        val_losses.append(np.mean(val_loss))
        train_accs.append(get_accuracy(net, ds_train))
        val_accs.append(get_accuracy(net, ds_val))

        np.save(f'{path}/val_loss_{model_file}', val_losses)
        np.save(f'{path}/train_loss_{model_file}', train_losses)
        np.save(f'{path}/val_acc_{model_file}', val_accs)
        np.save(f'{path}/train_acc_{model_file}', train_accs)


        if np.mean(val_loss) < best_val_loss:
          best_val_loss = np.mean(val_loss)
          torch.save(net.state_dict(), f'{path}/{model_file}')

        if verbose and epoch%freq==0:
          print('Epoch {}/{} || Loss:  Train {:.4f} | Validation {:.4f}'.format(
            epoch, epochs, np.mean(loss), np.mean(val_loss)))


train_anet(50, net, criterion, optimizer, train_loader,
      val_loader, catsdogs_train, catsdogs_val, scheduler,
      path='/usr/local/ML/alexnet_upd/l1', model_file='posalexnet_l1', positivate=True, l1alpha=1e-4)


