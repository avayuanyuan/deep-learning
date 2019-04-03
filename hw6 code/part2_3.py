import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#################################################################################

global  batch_size, n_class, n_z,gen_train

batch_size = 128
n_classes = 10
n_z = 100
num_epochs = 250
gen_train = 1

#################################################################################
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

    def forward(self, x, extract_features=0 ):
        conv1 = self.conv1(x)
        x = F.leaky_relu(self.ln1(conv1),negative_slope=0.1)
        conv2 = self.conv2(x)
        x = F.leaky_relu(self.ln2(conv2),negative_slope=0.1)
        x = self.conv_drop(x)
        conv3 = self.conv3(x)
        x = F.leaky_relu(self.ln3(conv3),negative_slope=0.1)
        conv4 = self.conv4(x)
        x = F.leaky_relu(self.ln4(conv4),negative_slope=0.1)
        x = self.conv_drop(x)
        conv5 = self.conv5(x)
        x = F.leaky_relu(self.ln5(conv5),negative_slope=0.1)
        conv6 = self.conv6(x)
        x = F.leaky_relu(self.ln6(conv6),negative_slope=0.1)
        x = self.conv_drop(x)
        conv7 = self.conv7(x)
        x = F.leaky_relu(self.ln7(conv7),negative_slope=0.1)
        conv8 = self.conv8(x)
        x = F.leaky_relu(self.maxpool( self.ln8(conv8) ),negative_slope=0.1)
        features = [conv1, conv2, conv3, conv4, conv5, conv6, conv7,conv8]
        feature_num = [32*32*196, 16*16*196,16*16*196, 8*8*196,8*8*196,8*8*196, 8*8*196, 4*4*196]
        
        x = x.view(-1,196)
        score = self.fc1(x)
        label = self.fc10(x)
        if extract_features >0 :
            index = int(extract_features-1)
            h = F.max_pool2d(features[index],4,4)
            h = h.view(-1, int(feature_num[index]/(4*4)))
            return score,label,h
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
        self.conv8 = nn.Conv2d(196,3,3,padding=1,stride=1)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = x.view(-1,196,4,4)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.tanh(self.conv8(x))
        
        return x

## some functions ###################################################################################
def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig

#######################################################################################################

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
testloader = enumerate(testloader)

model1 = torch.load('cifar10.model',map_location='cpu')
model1.cuda()
model1.eval()

model2 = torch.load('discriminator.model',map_location='cpu')
model2.cuda()
model2.eval()


######################################################################################################
batch_idx, (X_batch, Y_batch) = testloader.__next__()
X_batch = Variable(X_batch,requires_grad=True) .cuda()
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate) .cuda()
Y_batch = Variable(Y_batch).cuda()
##############################################################

def print_features(model,X_batch,extract_feature, name):

    X = X_batch.mean(dim=0)
    X = X.repeat(batch_size,1,1,1)
    
    Y = torch.arange(batch_size).type(torch.int64)
    Y = Variable(Y).cuda()
    
    lr = 0.1
    weight_decay = 0.001
    for i in range(200):
        _,output1,output = model(X,extract_feature)
    
        loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
        gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                  grad_outputs=torch.ones(loss.size()).cuda(),
                                  create_graph=True, retain_graph=False, only_inputs=True)[0]
    
        
        prediction = output1.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
        print(i,accuracy)
    
        X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
        X[X>1.0] = 1.0
        X[X<-1.0] = -1.0
        del loss, gradients
    ## save new images
    samples = X.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0,2,3,1)
    
    fig = plot(samples[0:100])
    plt.savefig('visualization/'+str(name)+'_'+str(extract_feature)+'.png', bbox_inches='tight')
    plt.close(fig)
    return 
    
print_features(model1,X_batch,2,"max_features_without")
print_features(model1,X_batch,4,"max_features_without")
print_features(model1,X_batch,6,"max_features_without")
print_features(model1,X_batch,8,"max_features_without")

print_features(model2,X_batch,2,"max_features_with")
print_features(model2,X_batch,4,"max_features_with")
print_features(model2,X_batch,6,"max_features_with")
print_features(model2,X_batch,8,"max_features_with")


