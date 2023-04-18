import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import torch.optim as optim


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

transform = transforms.Compose(
[transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 128

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

class Res2Net_BasicBlock_se(nn.Module):
    def __init__(self, in_channels, out_channels, stride, ratio=4, scales=4, downsample : Optional[nn.Module] = None) :
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        width = out_channels // scales
        self.width = width
        self.scales = scales
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.convs2 = nn.ModuleList(
            [nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=1, padding=1, bias=False)
             for _ in range(scales-1)])
        
        self.bns2 = nn.ModuleList(
            [nn.BatchNorm2d(width) for _ in range(scales-1)]
        )
        self.downsample = downsample

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels // ratio, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels= out_channels // 4, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        _x = x
        out = self.relu(self.bn1(self.conv1(x)))

        split_x = torch.split(out, self.width, 1)
        split_y = []
        for i in range(self.scales):
            if i == 0:
                split_y.append(split_x[0])
            elif i == 1:
                split_y.append(self.relu(self.bns2[i-1](self.convs2[i-1](split_x[i]))))
            else:
                split_y.append(self.relu(self.bns2[i-1](self.convs2[i-1](split_x[i] + split_y[i-1]))))
        out = torch.cat(split_y, 1)
        coefficient = self.se(out)
        if self.downsample is not None:
            _x = self.downsample(_x)

        out = F.relu(_x + out*coefficient)
        
        return out

# 4 3 32 32

class Res2Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # 3 32 32 -> 64 32 32
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2_1 = Res2Net_BasicBlock_se(64, 64, 1)
        self.conv2_2 = Res2Net_BasicBlock_se(64, 64, 1)

        self.conv3_1 = Res2Net_BasicBlock_se(64, 128, 2,
                                downsample=nn.Sequential(
                                    nn.Conv2d(64, 128, 1, 2, bias=False),
                                    nn.BatchNorm2d(128)
                                ))
        # 64 32 32 -> 128, 16, 16
        self.conv3_2 = Res2Net_BasicBlock_se(128, 128, 1)

        self.conv4_1 = Res2Net_BasicBlock_se(128, 256, 2,
                                downsample=nn.Sequential(
                                    nn.Conv2d(128, 256, 1, 2, bias=False),
                                    nn.BatchNorm2d(256)
                                ))
        # 128, 16, 16 -> 256, 8, 8
        self.conv4_2 = Res2Net_BasicBlock_se(256, 256, 1)

        self.conv5_1 = Res2Net_BasicBlock_se(256, 512, 2, 
                                downsample=nn.Sequential(
                                    nn.Conv2d(256, 512, 1, 2, bias=False),
                                    nn.BatchNorm2d(512)
                                  ))
        # 256, 8, 8 -> 512, 4, 4
        self.conv5_2 = Res2Net_BasicBlock_se(512, 512, 1)

        #self.conv5_2 = BasicBlock(512, 512, 1)
        #512,2,2
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, num_classes)
        #self.fc1 = nn.Linear(512, 100)
        #self.fc2 = nn.Linear(100, 10)

        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):       
        x = self.conv1(x)
        x = self.relu(self.bn1(x))

        x = self.conv2_1(x)
        x= self.conv2_2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        x = self.fc(x)
        #x = self.relu(self.fc1(x))
        #x = F.softmax(self.fc2(x), dim = 1)
        #x = F.softmax(x, dim=1)
        return x

net = Res2Net()
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
        if i % 25 == 24:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 25:.3f}')
            running_loss = 0.0
            
if __name__ == "__main__":

    lossv,accv=[],[]

    print("Begin training")
    for epoch in range(35):  # loop over the dataset multiple times
        train(epoch)
        test(lossv,accv)
    print('Finished Training')

    PATH = '/codes/DeepLearning/DL/CNN/cifar_resnet_35.pth'
    torch.save(net.state_dict(), PATH)