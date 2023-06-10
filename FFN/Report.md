# FFN

2011269 王楠舟

## 原始版本MLP

### 网络结构

原始版本MLP网络结构如下所示：

```python
Net(
  (fc1): Linear(in_features=784, out_features=100, bias=True)
  (fc1_drop): Dropout(p=0.2, inplace=False)
  (fc2): Linear(in_features=100, out_features=80, bias=True)
  (fc2_drop): Dropout(p=0.2, inplace=False)
  (fc3): Linear(in_features=80, out_features=10, bias=True)
)
```

可见该MLP前馈神经网络架构主要由三个全连接层构成，分别从28*28维的图像空间映射到100维、100维映射到80维、80维映射到10维的数字类型空间。为了避免训练过程的过拟合，在全连接层之间通过Dropout随机丢弃部分数据。

### 训练结果

训练过程使用SGD随机梯度下降，初始化学习率为0.01

**损失曲线：**

<img src="img\b1.png" alt="b1" style="zoom: 67%;" />

**准确率曲线：**

<img src="img\b2.png" alt="b2" style="zoom: 67%;" />

**最终模型在测试集上的准确率为96.98%**

## 参数改进

### epoch

首先在损失值和精确度曲线上能直观发现，迭代训练10次后模型的损失值和精确度曲线没有完全收敛，所以最直观的改进就是增大epoch，因为MNIST数据集很小，模型容易出现过拟合，所以适当增大尝试epoch=15，结果如下：

![image-20230610153023453](E:\DeepLearning\DL\FFN\img\image-20230610153023453.png)

![image-20230610153054086](E:\DeepLearning\DL\FFN\img\image-20230610153054086.png)

**最终模型在测试集上的准确率为97.58%，可见模型在epoch=10时没有完全收敛，增大epoch可以提升准确率。**

### Optimizer

训练中使用的是最简单的SGD，尝试使用Adam比较其和SGD收敛速度：

**Adam(lr=0.01):**

<img src="E:\DeepLearning\DL\FFN\img\image-20230610154404231.png" alt="image-20230610154404231" style="zoom:67%;" />

使用Adam后我们发现，一开始收敛速度要快于SGD很多，然而后续的训练阶段难以优化并达到最优解，是因为Adam使用自适应学习率，可以根据每个参数的梯度自动调整学习率。然而，如果学习率设置得太高，可能会导致训练过程中的震荡或不稳定。这可能使得模型无法收敛到最优解，而是在局部最小值附近徘徊，所以我们降低初始化学习率：

**Adam(lr=0.001):**

<img src="C:\Users\LEGION\AppData\Roaming\Typora\typora-user-images\image-20230610155101862.png" alt="image-20230610155101862" style="zoom:67%;" />

**可见使用Adam优化器在降低初始化化学习率后，模型收敛速度更快了，而且震荡现象明显减小，最终在测试集上的准确率为97.76%**

### 正则化

使用Adam优化器后明显看到模型抖动，所以在此基础上对模型的权重应用L2正则化：

**Adam(model.parameters(), lr=0.001, weight_decay=1e-4)：**

<img src="E:\DeepLearning\DL\FFN\img\image-20230610160513091.png" alt="image-20230610160513091" style="zoom:67%;" />

**在应用正则化后模型训练过程损失值下降更加平滑，虽然还有一点抖动但不明显，最终在测试集上准确率为97.71%**

### MLP参数

在第一层全连接层由100维改为300维，原因是图像特征从原来的784维度降为100维可能会丢失过多图像信息，适当增大隐藏层能增大模型参数，学习更多图像信息：

```
Net(
  (fc1): Linear(in_features=784, out_features=300, bias=True)
  (fc1_drop): Dropout(p=0.2, inplace=False)
  (fc2): Linear(in_features=300, out_features=80, bias=True)
  (fc2_drop): Dropout(p=0.2, inplace=False)
  (fc3): Linear(in_features=80, out_features=10, bias=True)
)
```

<img src="E:\DeepLearning\DL\FFN\img\image-20230610161521531.png" alt="image-20230610161521531" style="zoom:67%;" />

**最终在测试集上准确率为98.06%，说明增加隐藏层参数确实能优化模型**

我们再尝试加深模型，尝试四层MLP网络能否由于三层MLP，但是不希望加深网络会增加过大参数，所以选择后面添加一层全连接层：

```
Net(
  (fc1): Linear(in_features=784, out_features=300, bias=True)
  (fc1_drop): Dropout(p=0.2, inplace=False)
  (fc2): Linear(in_features=300, out_features=100, bias=True)
  (fc2_drop): Dropout(p=0.2, inplace=False)
  (fc3): Linear(in_features=100, out_features=30, bias=True)
  (fc3_drop): Dropout(p=0.2, inplace=False)
  (fc4): Linear(in_features=30, out_features=10, bias=True)
)
```

<img src="E:\DeepLearning\DL\FFN\img\image-20230610162635145.png" alt="image-20230610162635145" style="zoom:67%;" />

**最终四层MLP结构在测试集上准确率为98.22%**

## 拓展-MLPMixer

MLPMixer通过多层感知机来对图像中的特征进行提取和组合，它通过堆叠多个MLP块来构建网络，每个MLPMixer基础块中包含一个token_mlp和channel_mlp，前者用来交换图像不同位置的信息，后者来学习同位置不同通道的信息。

代码实现：

```python
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.2)
        )
    def forward(self, x):
        return self.net(x)

class MLPMixerBlock(nn.Module):
    def __init__(self, dim, num_patches, token_dim, channel_dim):
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patches, token_dim),
            Rearrange('b d n -> b n d')
        )
        self.channel_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim)
        )

    def forward(self, x):
        x = x + self.token_mlp(x)
        x = x + self.channel_mlp(x)
        return x
        

class MLPMixer(nn.Module):
    def __init__(self, image_size, channels, patch_size, dim, depth, token_dim, channel_dim):
        super(MLPMixer, self).__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) **2
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MLPMixerBlock(dim, self.num_patches, token_dim, channel_dim))
        
        self.ln = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, 10)
    
    def forward(self, x):
        x = self.patch_embedding(x)

        for block in self.mixer_blocks:
            x = block(x)

        x = self.ln(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
   
model = MLPMixer(image_size=28, channels=1, patch_size=7, dim=32, depth=3, token_dim=32, channel_dim=128).to(device)
```

最终我实现一个了三层的MLPMixer网络，下面展示其中一层的具体信息：

```
(mixer_blocks): ModuleList(
    (0): MLPMixerBlock(
      (token_mlp): Sequential(
        (0): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (1): Rearrange('b n d -> b d n')
        (2): FeedForward(
          (net): Sequential(
            (0): Linear(in_features=16, out_features=32, bias=True)
            (1): GELU()
            (2): Dropout(p=0.2, inplace=False)
            (3): Linear(in_features=32, out_features=16, bias=True)
            (4): Dropout(p=0.2, inplace=False)
          )
        )
        (3): Rearrange('b d n -> b n d')
      )
      (channel_mlp): Sequential(
        (0): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (1): FeedForward(
          (net): Sequential(
            (0): Linear(in_features=64, out_features=128, bias=True)
            (1): GELU()
            (2): Dropout(p=0.2, inplace=False)
            (3): Linear(in_features=128, out_features=64, bias=True)
            (4): Dropout(p=0.2, inplace=False)
          )
        )
      )
    )
```

**实验结果：**

<img src="E:\DeepLearning\DL\FFN\img\image-20230610190646164.png" alt="image-20230610190646164" style="zoom:67%;" />



**最终MLPMixer在测试集上准确率为98.31%**

