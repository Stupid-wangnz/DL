# FFN

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

原始MLP前馈神经网络架构主要由三个全连接层构成，分别从28*28维的图像空间映射到100维、100维映射到80维、80维映射到10维的数字类型空间。为了避免训练过程的过拟合，在全连接层之间通过Dropout随机丢弃部分数据，丢弃概率为20%。

### 训练结果

**损失曲线**

<img src="img\b1.png" alt="b1" style="zoom:60%;" />

**准确率曲线**

<img src="img\b2.png" alt="b2" style="zoom:67%;" />

最终模型在测试集上的准确率为97.09%。

## 改进

### epoch

首先在损失值和精确度曲线上能直观发现，迭代训练10次后模型的损失值和精确度曲线没有到达收敛，所以最直观的改进就是增大epoch，当然也不能盲目增大，不然容易出现过拟合，所以适当将迭代次数增大，尝试epoch=12。

**epoch = 12	Average loss = 0.867	Accuracy = 97.28%**

<img src="img\image-20230321203324448.png" alt="image-20230321203324448" style="zoom:67%;" />

<img src="img\image-20230321203431222.png" alt="image-20230321203431222" style="zoom:67%;" />

### Learning rate

在训练阶段，我们发现在训练迭代五六轮后，在训练集上的损失值波动比较大，推测是到训练到后面模型参数微调学习率过大，所以将学习率调小并适当增大迭代次数。

