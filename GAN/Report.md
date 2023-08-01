# GAN

## 基础网络

### 网络结构：

```
Discriminator(
  (fc1): Linear(in_features=784, out_features=128, bias=True)
  (nonlin1): LeakyReLU(negative_slope=0.2)
  (fc2): Linear(in_features=128, out_features=1, bias=True)
)
Generator(
  (fc1): Linear(in_features=100, out_features=128, bias=True)
  (nonlin1): LeakyReLU(negative_slope=0.2)
  (fc2): Linear(in_features=128, out_features=784, bias=True)
)
```

### 训练结果：

**判断器对真图像和假图像的判别能力：**

<img src="img\image-20230620170532495.png" alt="image-20230620170532495" style="zoom:67%;" />

<img src="img\image-20230620170644470.png" alt="image-20230620170644470" style="zoom:67%;" />

**判别器和生成器的损失值：**

<img src="img\image-20230620170736330.png" alt="image-20230620170736330" style="zoom:67%;" />

可以看到GAN在训练过程中判别器和生成器的损失值都有较大抖动，可见对抗网络较难训练，但也能发现总体趋势是判别器损失值变大，而生成器损失值变小。

### 自定义随机数：

随机自定义一组随机数生成8张图片结果：

![image-20230620173956942](img\image-20230620173956942.png)

可以见到生成图像较为模糊。有大致轮廓但细致内容不清楚。

**针对自定义的100个随机数，自由挑选5个随机数，查看调整每个随机数时，生成图像的变化：**

在这里，没三行图像为一组，每一组分别控制第5、25、45、65、85个随机数，分别控制为-3、0、3，观察结果：

<img src="img\image-20230620173804814.png" alt="image-20230620173804814" style="zoom:67%;" />

**第一组图像可见：调整第5个随机数的时候，主要控制了图像的上部分的内容，小于0时上部分为空，大于0时上部分内容逐渐明亮；**

**第二组图像：第25个随机数主要影响图像轮廓大小，当小于0的时候整体内容轮廓较小；**

**第三组图像：第45个随机数主要影响图像内容的明亮，当大于0的时候图像整体更加明亮；**

**第四组图像：第65个随机数同样也主要图像中心的内容，越接近0的时候图像中心更清晰；**

**另一个较为明显的例子调整第5组图像，调整第85个随机数的时候，控制的是图像的四周部分，小于0是为暗，大于0时图像四周逐渐出现内容。**

由此可见，在生成器的输入中，每一个位置上的输入控制了图像中不同部分的明暗，从而需要由许多随机数一起组合来控制图像的具体内容，以上面第三组图像和第五组图像中的第二张生成的图像为例，一个更像鞋子，一个更像外套，可见需要多组随机数组合一起才能控制整张图像内容，单独一个随机数对图像改变并不大。

## 用卷积实现生成器和判别器：DCGAN

### 自己实现的DCGAN代码：

**判别器中就是简单的卷积层，最后激活函数的Sigmoid来判别输入是否是真实图像；在生成器中使用逆卷积层，将输入逐步放大，最终放大到图像大小。**

注意在这里为了代码实现，将FashionMNIST图像大小放大到32*32.

```python
class CNNDiscriminator(nn.Module):
    def __init__(self, features=64):
        super(CNNDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features, features*2, 4, 2, 1),
            nn.BatchNorm2d(features*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features*2, features*4, 4, 2, 1),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features*4, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )
    def forward(self, img):
        validity = self.model(img)
        return validity.view(-1, 1)

class CNNGenerator(nn.Module):
    def __init__(self, latent_dim=100, features=64):
        super(CNNGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, features*8, 4, 1, 0),
            nn.ReLU(True),

            # Transpose block 3
            nn.ConvTranspose2d(features*8, features*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*4),
            nn.ReLU(),

            # Transpose block 4
            nn.ConvTranspose2d(features*4, features*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*2),
            nn.ReLU(),

            # Last transpose block (different)
            nn.ConvTranspose2d(features*2, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        img = self.model(z)
        return img
```

### 训练结果：

<img src="img\img.png" alt="img" style="zoom:67%;" />

<img src="img\img1.png" alt="img" style="zoom:67%;" />

<img src="img\img2.png" alt="img" style="zoom:67%;" />

对比生成器和判别器的损失函数可以发现，判别器的损失值一直较小，而生成器损失在震荡中逐渐的增大，可见GAN还是非常难以训练。

### 自定义随机数：

可见，DCGAN生成的图像更加清晰，细节更加具体，但是有部分图像完全扭曲，可能是迭代次数太少了。

![img](img\img3.png)

**针对自定义的100个随机数，自由挑选5个随机数，查看调整每个随机数时，生成图像的变化：**

<img src="img\image-20230620213820299.png" alt="image-20230620213820299" style="zoom:67%;" />

对比于简单的GAN网络，DCGAN生成的图像更加完整且清晰。可以清楚的发现，和前面分析的GAN一样，同样是由不同随机数来控制不同区域，从而组合生成一幅完整的图像，但是DCGAN不同随机数的影响更大。

**如第一组图像中，该随机数控制的是图像左侧区域，在衣物中就是左侧袖子的长度，在最后一幅图像中能清晰发现，随机数越大，衣服的袖子越小，所以对于衣服而言，该随机数较小能生成更容易欺骗判别器的图像**；

**在第四组图像中能发现该随机数对生成物体的影响非常大，当随机数较小时还是正常衣物的形状，但是当值较大的时候生成的图像明显不符合衣物的形状，可见DCGAN对随机数比较敏感**；

**在第五组图像中，该随机数控制的则是整体物件的长度，可以发现随机数越小，生成图像中的衣服越修长，随机数越大，生成图像中的衣物越宽大，这是在前面实现的GAN中没有发现的。**



