from enum import Flag
from numpy import block
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import torch.optim as optim

import torchvision.models.densenet
torch.cuda.empty_cache()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

transform = transforms.Compose(
[transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 32

trainset = torchvision.datasets.CIFAR10(root='/codes/DeepLearning/DL/CNN/data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/codes/DeepLearning/DL/CNN/data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class DenseLayer(nn.Module):
    def __init__(self, input_size, g, bn_size=4, drop_rate=0.2):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(input_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_size, out_channels=g*bn_size, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(g*bn_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=g*bn_size, out_channels=g, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        _x = self.layer(x)
        _x = self.drop(_x)
        _x = torch.concat([x, _x], 1)
        return _x
    
class DenseBlock(nn.Module):
    def __init__(self, layers, input_size, g, bn_size=4, drop_rate=0.2):
        super().__init__()
        self.block = nn.ModuleList([])
        for i in range(layers):
            self.block.append(DenseLayer(input_size + i * g, g, bn_size, drop_rate))
    
    def forward(self, x):
        for layer in self.block:
            x = layer(x)
        return x
    
# 32*32 -> 32*32 -> 16*16 -> 8*8 -> 4*4
class DenseNet(nn.Module):
    def __init__(self, first_features, g=32, bn_size=4, drop_rate=0.2, num_classes=10):
        super().__init__()
        # not need the max pool
        self.conv_ = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=first_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(first_features),
            nn.ReLU(inplace=True)
        )
        self.blocks_transition = nn.ModuleList([]) 
        #DenseNet121
        layer_args = (6, 12, 24, 16)
        features = first_features
        for i, layers in enumerate(layer_args):
            self.blocks_transition.append(DenseBlock(layers=layers, input_size=features, g=g, bn_size=bn_size, drop_rate=drop_rate))
            features += layers * g
            if i != len(layer_args)-1:
                self.blocks_transition.append(
                    nn.Sequential(
                        nn.BatchNorm2d(features),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels=features, out_channels=features // 2, kernel_size=1, stride=1, bias=False),
                        nn.AvgPool2d(2, stride=2)
                    )
                )
                #update feature in next block
                features = features // 2
        self.norm_relu = nn.Sequential(
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

        #fc layer
        self.fc = nn.Linear(features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        _x = self.conv_(x)
        for block in self.blocks_transition:
            _x = block(_x)
        #global avg pool
        _x = F.avg_pool2d(_x, 4, stride=1)
        _x = _x.view(_x.size(0), -1)
        out = self.fc(_x)
        return out


net = DenseNet(first_features=64, g=32, bn_size=4, drop_rate=0.2, num_classes=10)
model = net.to(device)
learn_rate = 0.001

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

def test(loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in testloader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(testloader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(testloader.dataset)
    accuracy_vector.append(accuracy)
    
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(testloader.dataset), accuracy))

def train(epoch):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0
            
if __name__ == "__main__":

    lossv,accv=[],[]

    print("Begin training")
    for epoch in range(50):  # loop over the dataset multiple times
        train(epoch)
        test(lossv,accv)
    print('Finished Training')

    PATH = '/codes/DeepLearning/DL/CNN/cifar_densenet.pth'
    torch.save(net.state_dict(), PATH)