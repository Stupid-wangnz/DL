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

<img src="E:\DeepLearning\DL\GAN\img\image-20230620170532495.png" alt="image-20230620170532495" style="zoom:67%;" />

<img src="E:\DeepLearning\DL\GAN\img\image-20230620170644470.png" alt="image-20230620170644470" style="zoom:67%;" />

**判别器和生成器的损失值：**

<img src="E:\DeepLearning\DL\GAN\img\image-20230620170736330.png" alt="image-20230620170736330" style="zoom:67%;" />

可以看到GAN在训练过程中判别器和生成器的损失值都有较大抖动，可见对抗网络较难训练，但也能发现总体趋势是判别器损失值变大，而生成器损失值变小。

### 自定义随机数：

随机自定义一组随机数生成8张图片结果：

![image-20230620173956942](E:\DeepLearning\DL\GAN\img\image-20230620173956942.png)

可以见到生成图像较为模糊。有大致轮廓但细致内容不清楚。

**针对自定义的100个随机数，自由挑选5个随机数，查看调整每个随机数时，生成图像的变化：**

在这里，没三行图像为一组，每一组分别控制第5、25、45、65、85个随机数，分别控制为-3、0、3，观察结果：

<img src="E:\DeepLearning\DL\GAN\img\image-20230620173804814.png" alt="image-20230620173804814" style="zoom:67%;" />

**第一组图像可见：调整第5个随机数的时候，主要控制了图像的上部分的内容，小于0时上部分为空，大于0时上部分内容逐渐明亮；**

**第二组图像：第25个随机数主要影响图像轮廓大小，当小于0的时候整体内容轮廓较小；**

**第三组图像：第45个随机数主要影响图像内容的明亮，当大于0的时候图像整体更加明亮；**

**第四组图像：第65个随机数同样也主要图像中心的内容，越接近0的时候图像中心更清晰；**

**另一个较为明显的例子调整第5组图像，调整第85个随机数的时候，控制的是图像的四周部分，小于0是为暗，大于0时图像四周逐渐出现内容。**

由此可见，在生成器的输入中，每一个位置上的输入控制了图像中不同部分的明暗，从而需要由许多随机数一起组合来控制图像的具体内容，以上面第三组图像和第五组图像中的第二张生成的图像为例，一个更像鞋子，一个更像外套，可见需要多组随机数组合一起才能控制整张图像内容，单独一个随机数对图像改变并不大。



<img src="E:\DeepLearning\DL\GAN\img\image-20230620161255073.png" alt="image-20230620161255073" style="zoom:67%;" />

<img src="E:\DeepLearning\DL\GAN\img\image-20230620161418413.png" alt="image-20230620161418413" style="zoom:67%;" />

<img src="E:\DeepLearning\DL\GAN\img\image-20230620161230404.png" alt="image-20230620161230404" style="zoom:67%;" />



![image-20230611000228350](E:\DeepLearning\DL\GAN\img\image-20230611000228350.png)