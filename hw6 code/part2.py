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

global batch_size, n_class, n_z,gen_train

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

    def forward(self, x ):
        
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

model = torch.load('cifar10.model',map_location='cpu')
model.cuda()
model.eval()
######################################################################################################
batch_idx, (X_batch, Y_batch) = testloader.__next__()
X_batch = Variable(X_batch,requires_grad=True) .cuda()
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).cuda()
Y_batch = Variable(Y_batch).cuda()

samples = X_batch.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/real_images.png', bbox_inches='tight')
plt.close(fig)


_,output = model(X_batch)
prediction = output.data.max(1)[1] # first column has actual prob.
accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
print(accuracy)


##lightly jitter all input images
criterion = nn.CrossEntropyLoss(reduce=False)
loss = criterion(output, Y_batch_alternate)

gradients = torch.autograd.grad(outputs=loss, inputs=X_batch,
                          grad_outputs=torch.ones(loss.size()).cuda(),
                          create_graph=True, retain_graph=False, only_inputs=True)[0]

# save gradient jitter
gradient_image = gradients.data.cpu().numpy()
gradient_image = (gradient_image - np.min(gradient_image))/(np.max(gradient_image)-np.min(gradient_image))
gradient_image = gradient_image.transpose(0,2,3,1)
fig = plot(gradient_image[0:100])
plt.savefig('visualization/gradient_image.png', bbox_inches='tight')
plt.close(fig)



## tter input image
gradients[gradients>0.0] = 1.0
gradients[gradients<0.0] = -1.0

gain = 8.0
X_batch_modified = X_batch - gain*0.007843137*gradients
X_batch_modified[X_batch_modified>1.0] = 1.0
X_batch_modified[X_batch_modified<-1.0] = -1.0

## evaluate new fake images
_, output = model(X_batch_modified)
prediction = output.data.max(1)[1] # first column has actual prob.
accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
print(accuracy)

## save fake images
samples = X_batch_modified.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/jittered_images.png', bbox_inches='tight')
plt.close(fig)

###################################################################


