import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import torch.optim as optim

<<<<<<< Updated upstream
torchvision.models.resnet18()
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

=======
>>>>>>> Stashed changes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 4 3 32 32

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample : Optional[nn.Module] = None) :
        super().__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        _x = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            _x = self.downsample(x)

        out = self.relu(_x + out)
        
        return out
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2_1 = BasicBlock(64, 64, 1)
        self.conv2_2 = BasicBlock(64, 64, 1)

        self.conv3_1 = BasicBlock(64, 128, 2,
<<<<<<< Updated upstream
                                downsample=LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 128//4, 128//4), "constant", 0))
                                )
        self.conv3_2 = BasicBlock(128, 128, 1)

        self.conv4_1 = BasicBlock(128, 256, 2,
                                downsample=LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 256//4, 256//4), "constant", 0))
                                )
=======
                                downsample=nn.Sequential(
                                    nn.Conv2d(64, 128, 1, 2, bias=False),
                                    nn.BatchNorm2d(128)
                                ))
        self.conv3_2 = BasicBlock(128, 128, 1)

        self.conv4_1 = BasicBlock(128, 256, 2,
                                downsample=nn.Sequential(
                                    nn.Conv2d(128, 256, 1, 2, bias=False),
                                    nn.BatchNorm2d(256)
                                ))
>>>>>>> Stashed changes
        self.conv4_2 = BasicBlock(256, 256, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)
    def forward(self, x):       
        x = self.conv1(x)
        x = self.relu(self.bn1(x))

        x = self.conv2_1(x)
        x= self.conv2_2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)

<<<<<<< Updated upstream
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc(x)
        return x

net = ResNet()
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

torchvision.models.resnet18()
=======
        x = self.conv4_1(x)
        x = self.conv4_2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc(x)
        return x
>>>>>>> Stashed changes
