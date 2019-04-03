from time import time
from datetime import datetime
from os.path import isfile

from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms


import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#################################################################################

global rootdir, batch_size

rootdir = '/u/training/tra385/scratch/hw6'
batch_size = 128

## Models ###############################################################################

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 196, 3,padding=1)
        self.ln1 = nn.LayerNorm([196,32,32])
        self.conv2 = nn.Conv2d(196, 196, 3,padding=1,stride=2)
        self.ln2 = nn.LayerNorm([196,16,16])
        self.conv3 = nn.Conv2d(196, 196, 3,padding=1)
        self.ln3 = nn.LayerNorm([196,16,16])
        self.conv4 = nn.Conv2d(196, 196, 3,padding=1,stride=2)
        self.ln4 = nn.LayerNorm([196,8,8])
        self.conv5 = nn.Conv2d(196, 196,3,padding=1)
        self.ln5 = nn.LayerNorm([196,8,8])
        self.conv6 = nn.Conv2d(196,196,3,padding=1)
        self.ln6 = nn.LayerNorm([196,8,8])
        self.conv7 = nn.Conv2d(196,196,3,padding=1)
        self.ln7 = nn.LayerNorm([196,8,8])
        self.conv8 = nn.Conv2d(196,196,3,padding=1,stride=2)
        self.ln8 = nn.LayerNorm([196,4,4])
        self.maxpool = nn.MaxPool2d(4,4)

        self.conv_drop = nn.Dropout(p=0.25)
     
        self.fc1 = nn.Linear(196,1)
        self.fc10 = nn.Linear(196,10)

    def forward(self, x):
        x = F.leaky_relu(self.ln1(self.conv1(x)),negative_slope=0.1)
        x = F.leaky_relu(self.ln2(self.conv2(x)),negative_slope=0.1)
        x = self.conv_drop(x)
        x = F.leaky_relu(self.ln3(self.conv3(x)),negative_slope=0.1)
        x = F.leaky_relu(self.ln4(self.conv4(x)),negative_slope=0.1)
        x = self.conv_drop(x)
        x = F.leaky_relu(self.ln5(self.conv5(x)),negative_slope=0.1)
        x = F.leaky_relu(self.ln6(self.conv6(x)),negative_slope=0.1)
        x = self.conv_drop(x)
        x = F.leaky_relu(self.ln7(self.conv7(x)),negative_slope=0.1)
        x = F.leaky_relu(self.maxpool( self.ln8(self.conv8(x)) ),negative_slope=0.1)
        
        x = x.view(-1,196)
        score = self.fc1(x)
        label = self.fc10(x)
        return score,label

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100,196*4*4)
        self.conv1 = nn.ConvTranspose2d(196,196,4,padding=1,stride=2)      
        self.bn1 = nn.BatchNorm2d(196)
        self.conv2 = nn.Conv2d(196, 196, 3,padding=1,stride=1)
        self.bn2 = nn.BatchNorm2d(196)
        self.conv3 = nn.Conv2d(196, 196, 3,padding=1,stride=1)
        self.bn3 = nn.BatchNorm2d(196)
        self.conv4 = nn.Conv2d(196, 196, 3,padding=1,stride=1)
        self.bn4 = nn.BatchNorm2d(196)
        self.conv5 = nn.ConvTranspose2d(196,196,4,padding=1,stride=2)      
        self.bn5 = nn.BatchNorm2d(196)
        self.conv6 = nn.Conv2d(196,196,3,padding=1)
        self.bn6 = nn.BatchNorm2d(196)
        self.conv7 = nn.ConvTranspose2d(196,196,4,padding=1,stride=2)      
        self.bn7 = nn.BatchNorm2d(196)
        self.conv8 = nn.Conv2d(196,196,3,padding=1,stride=1)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.conv8(x))
        
        return x

##################################################################################################


def get_accu(model,testloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
         for data in testloader:
             images, labels = data
             images = torch.autograd.Variable(images).cuda()
             labels = torch.autograd.Variable(labels).cuda()
             _, outputs = model(images)
             _, predicted = torch.max(outputs.data, 1)
             total += labels.size(0)
             correct += (predicted == labels).sum().item()
    return 100.0*correct/total


#################################################################################################
 
print ( datetime.now().strftime("%y-%m-%d %H:%M:%S"))
print ("Loading Data")  
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root=rootdir+'/data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root=rootdir+'/data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
print ("Done")

## Initialization #################################################################################

print ( datetime.now().strftime("%y-%m-%d %H:%M:%S"))
print ("Initialization")
model =  Discriminator()
model.cuda()
criterion = nn.CrossEntropyLoss()
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print ("Done")

average_time = 0
for epoch in range(100):
    total_cost = 0
    start_time = time()
    model.train()
    if(epoch==50):
         for  param_group in optimizer.param_groups:
              param_group['lr'] = learning_rate/10.0
    if(epoch==75):
         for  param_group in optimizer.param_groups:
              param_group['lr'] = learning_rate/100.0

    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

         if(Y_train_batch.shape[0] < batch_size):
              continue

         for group in optimizer.param_groups:
             for p in group['params']:
                  state = optimizer.state[p]
                  if('step' in state and state['step']>=1024):
                      state['step'] = 1000

         X_train_batch = torch.autograd.Variable(X_train_batch).cuda()
         Y_train_batch = torch.autograd.Variable(Y_train_batch).cuda()
         _, output = model(X_train_batch)

         loss = criterion(output, Y_train_batch)
         optimizer.zero_grad()
         total_cost+=loss.item()

         loss.backward()
         optimizer.step()
    end_time = time()
    average_time = average_time*epoch + end_time-start_time
    average_time = average_time/(epoch+1)
    average_cost = total_cost/float(batch_idx+1)
    if epoch%8 ==0:
         hours, rem = divmod(average_time, 3600)
         minutes, seconds = divmod(rem, 60)
         seconds = int(round(seconds))
         average_time_str = "{:0>2}:{:0>2}:{:0>2}".format(int(hours),int(minutes),seconds)
         test_accu = get_accu(model,testloader)
         #train_accu = get_accu(model,trainloader)
         print (str(epoch).zfill(2),"{:+.4E}".format(average_cost), \
                "{:+.1f}".format(test_accu), average_time_str)

torch.save(model,rootdir+'/cifar10.model')
