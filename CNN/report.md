## 基础网络

### 基础网络结构

```
Net(
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
```

### 训练结果：

<img src="E:\DeepLearning\DL\CNN\img\image-20230620103954044.png" alt="image-20230620103954044" style="zoom:67%;" />

<img src="E:\DeepLearning\DL\CNN\img\image-20230620104013354.png" alt="image-20230620104013354" style="zoom:67%;" />

**最终在测试集上准确率为63.48%**

## 后续实验设置

因为Adam可能容易出现过拟合，在后续实验中使用的都是SGD优化器，配合MultiStepLR进行动态学习率调整。

**需要在Dataloader中对训练数据进行预处理，使用随机剪裁、随机翻转等数据增强手段，否则由于后续模型参数较大，容易在训练集上出现过拟合**

## ResNet

### **自己实现的ResNet代码：**

我实现了一个简化的ResNet代码，其中第一层卷积层卷积核大小由原来的7改为3，因为我们在CIFAR10数据集上训练，图像大小为32*32，卷积核大小不适合过大。

在ResNet18的基础上稍微进行了简化，实现了一个ResNet14的网络结构：

```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample= None) :
        super().__init__()

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

        out = F.relu(_x + out)
        
        return out

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2_1 = BasicBlock(64, 64, 1)
        self.conv2_2 = BasicBlock(64, 64, 1)

        self.conv3_1 = BasicBlock(64, 128, 2,
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

        x = self.conv4_1(x)
        x = self.conv4_2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc(x)
        return x
```

### 训练结果：

<img src="E:\DeepLearning\DL\CNN\img\image-20230612215057648.png" alt="image-20230612215057648" style="zoom:67%;" />

<img src="E:\DeepLearning\DL\CNN\img\image-20230612215127497.png" alt="image-20230612215127497" style="zoom:67%;" />

**最终在测试集上准确率为90.32%**

对各个类别的分类准确率：

```
Accuracy for class: plane is 89.8 %
Accuracy for class: car   is 94.9 %
Accuracy for class: bird  is 83.0 %
Accuracy for class: cat   is 77.9 %
Accuracy for class: deer  is 88.4 %
Accuracy for class: dog   is 81.0 %
Accuracy for class: frog  is 91.9 %
Accuracy for class: horse is 91.3 %
Accuracy for class: ship  is 93.8 %
Accuracy for class: truck is 91.9 %
```



## SEResNet

### 自己实现的SEResNet代码：

在上面实现的ResNet14上添加se：

```python
class BasicBlock_se(nn.Module):
    def __init__(self, in_channels, out_channels, stride, ratio=4, downsample=None) :
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels // ratio, 
                      kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels= out_channels // 4, out_channels=out_channels, 
                      kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        _x = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        coefficient = self.se(out)
        if self.downsample is not None:
            _x = self.downsample(x)

        out = F.relu(_x + out*coefficient)
        
        return out

class SE_ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # 3 32 32 -> 64 32 32
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2_1 = BasicBlock_se(64, 64, 1)
        self.conv2_2 = BasicBlock_se(64, 64, 1)

        self.conv3_1 = BasicBlock_se(64, 128, 2,
                                downsample=nn.Sequential(
                                    nn.Conv2d(64, 128, 1, 2, bias=False),
                                    nn.BatchNorm2d(128)
                                )
        self.conv3_2 = BasicBlock_se(128, 128, 1)
        
        self.conv4_1 = BasicBlock_se(128, 256, 2,
                                downsample=nn.Sequential(
                                    nn.Conv2d(128, 256, 1, 2, bias=False),
                                    nn.BatchNorm2d(256)
                                )
        self.conv4_2 = BasicBlock_se(256, 256, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(256, 10)

    def forward(self, x):       
        x = self.conv1(x)
        x = self.relu(self.bn1(x))

        x = self.conv2_1(x)
        x= self.conv2_2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        x = self.fc(x)
        
        return x
```

### 训练结果：

<img src="E:\DeepLearning\DL\CNN\img\image-20230612220215520.png" alt="image-20230612220215520" style="zoom:67%;" />

<img src="E:\DeepLearning\DL\CNN\img\image-20230612220228032.png" alt="image-20230612220228032" style="zoom:67%;" />

**最终在测试集上准确率为91.41%**

对各个类别的分类准确率：

```
Accuracy for class: plane is 90.4 %
Accuracy for class: car   is 94.8 %
Accuracy for class: bird  is 83.3 %
Accuracy for class: cat   is 81.5 %
Accuracy for class: deer  is 88.6 %
Accuracy for class: dog   is 83.1 %
Accuracy for class: frog  is 92.7 %
Accuracy for class: horse is 92.9 %
Accuracy for class: ship  is 94.2 %
Accuracy for class: truck is 90.7 %
```

## Res2Net

### 自己实现的Res2Net代码：

在上面实现的SE-ResNet基础上修改为Res2Net：

```
class Res2Net_BasicBlock_se(nn.Module):
    def __init__(self, in_channels, out_channels, stride, ratio=4, scales=4, downsample=None) :
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        width = out_channels // scales
        self.width = width
        self.scales = scales
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
        	stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.convs2 = nn.ModuleList(
            [nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, 
            	stride=1, padding=1, bias=False)
             for _ in range(scales-1)])
        
        self.bns2 = nn.ModuleList(
            [nn.BatchNorm2d(width) for _ in range(scales-1)]
        )
        self.downsample = downsample

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels // ratio, 
            	kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels= out_channels // 4, out_channels=out_channels, 
            	kernel_size=1, stride=1, bias=False),
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
                                )
        self.conv3_2 = Res2Net_BasicBlock_se(128, 128, 1)

        self.conv4_1 = Res2Net_BasicBlock_se(128, 256, 2,
                                downsample=nn.Sequential(
                                    nn.Conv2d(64, 128, 1, 2, bias=False),
                                    nn.BatchNorm2d(128)
                                )
        self.conv4_2 = Res2Net_BasicBlock_se(256, 256, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(256, num_classes)
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


        x = self.avgpool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        x = self.fc(x)
        return x
```

### 训练结果：

<img src="E:\DeepLearning\DL\CNN\img\image-20230612225644093.png" alt="image-20230612225644093" style="zoom:67%;" />

<img src="E:\DeepLearning\DL\CNN\img\image-20230612225705247.png" alt="image-20230612225705247" style="zoom:67%;" />

**最终在测试集上正确率为89.32%**

对各个类别的分类准确率：

```
Accuracy for class: plane is 86.3 %
Accuracy for class: car   is 92.7 %
Accuracy for class: bird  is 78.1 %
Accuracy for class: cat   is 69.7 %
Accuracy for class: deer  is 84.3 %
Accuracy for class: dog   is 75.1 %
Accuracy for class: frog  is 88.8 %
Accuracy for class: horse is 90.6 %
Accuracy for class: ship  is 91.9 %
Accuracy for class: truck is 90.8 %
```

Res2Net效果在CIFAR10上表现的一般，推测原因是CIFAR10数据集中图像中物体都比较大，Res2Net的多尺度表示特征没有明显的作用。

## DenseNet

### 自己实现的DenseNet代码：

```python
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
    def __init__(self, g=32, bn_size=4, drop_rate=0.2, num_classes=10):
        super().__init__()
        # not need the max pool
        self.conv_ = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.blocks_transition = nn.ModuleList([]) 
        #DenseNet121
        layer_args = (3, 6, 12, 8)
        features = 64
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
```

### 训练结果：

<img src="E:\DeepLearning\DL\CNN\img\image-20230620101524037.png" alt="image-20230620101524037" style="zoom:67%;" />

<img src="E:\DeepLearning\DL\CNN\img\image-20230620101544761.png" alt="image-20230620101544761" style="zoom:67%;" />

**最终在测试集上正确率为90.90%**

对各个类别的分类准确率：

```
Accuracy for class: plane is 88.1 %
Accuracy for class: car   is 94.9 %
Accuracy for class: bird  is 81.2 %
Accuracy for class: cat   is 77.0 %
Accuracy for class: deer  is 89.1 %
Accuracy for class: dog   is 80.2 %
Accuracy for class: frog  is 90.3 %
Accuracy for class: horse is 91.6 %
Accuracy for class: ship  is 94.8 %
Accuracy for class: truck is 91.1 %
```



## 没有跳跃连接的卷积网络、ResNet、DenseNet、SE-ResNet在训练过程中有什么不同

1. 没有跳跃连接的卷积网络：
   - 在没有跳跃连接的普通卷积网络中，每个层都通过卷积、池化等操作将输入逐层转换为输出，每个网络层的输出只传递给下一层作为输入，没有额外的直接连接或信息传递。
   - 没有跳跃连接的网络可能会受到梯度消失或梯度爆炸等问题的影响，限制了网络的深度和性能。
2. ResNet：
   - ResNet引入了跳跃连接或称为残差连接，它允许在网络中跨越多个层直接传递信息。
   - 残差连接通过将输入与层的输出相加，使得网络可以学习残差映射。这种结构使得梯度可以更容易地在网络中传播，有助于训练更深的网络，避免梯度消失和梯度爆炸的问题。
3. DenseNet：
   - DenseNet引入了密集连接，每个层的输出不仅传递给下一层，还通过短路连接传递给后续所有层。每个层接收来自前面所有层的特征图作为输入，而不仅仅是前一层的输出，从而增加了特征的多样性和信息的流动。
   - 密集连接有助于缓解梯度消失问题，提高信息传递效率，并且可以在不增加参数数量的情况下提高网络性能。但在实际运行中发现，DenseNet尽管不会增加参数，但是由于要保存低层的输出会导致占用过多显存。
4. SE-ResNet：
   - SE-ResNet在ResNet的基础上引入了注意力机制，以动态地调整每个特征通道的权重。
   - 通过引入Squeeze-and-Excitation模块，SE-ResNet可以自适应地学习特征通道的重要性，并加权地增强或抑制它们的表示能力。
   - SE-ResNet的注意力机制使得网络能够更加关注重要的特征通道，提高网络的表示能力和性能。

没有跳跃连接的卷积网络在训练过程中仅通过层之间的顺序连接进行信息传递，而ResNet通过残差连接解决了梯度消失和梯度爆炸的问题，SE-ResNet在ResNet的基础上引入了注意力机制来提高特征通道的表示能力，DenseNet通过密集连接增加了信息的流动和多样性。这些改进使得网络更深、更稳定，并且具有更强的表达能力。
