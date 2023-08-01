# RNN

2011269 王楠舟

## 原始版本RNN

**网络架构：**

```
RNN(
  (i2h): Linear(in_features=185, out_features=128, bias=True)
  (i2o): Linear(in_features=185, out_features=18, bias=True)
  (softmax): LogSoftmax(dim=1)
)
```

**实验结果：**

<img src="img\image-20230610215055180.png" alt="image-20230610215055180" style="zoom:67%;" />

<img src="img\image-20230610215117033.png" alt="image-20230610215117033" style="zoom:67%;" />

<img src="img\image-20230610215136034.png" alt="image-20230610215136034" style="zoom:67%;" />

最终准确率在63%左右，在预测矩阵图中能发现，部分语言的预测准确度很高，但一部分准确率很低，如German、Czech、Spanish等，导致整体准确率不高。

## LSTM实现

自己实现的LSTM代码：

```python
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.i_fgate = nn.Linear(input_size, hidden_size, bias=True)
        self.h_fgate = nn.Linear(hidden_size, hidden_size, bias=True)

        self.i_igate = nn.Linear(input_size, hidden_size, bias=True)
        self.h_igate = nn.Linear(hidden_size, hidden_size, bias=True)

        self.i_cell = nn.Linear(input_size, hidden_size, bias=True)
        self.h_cell = nn.Linear(hidden_size, hidden_size, bias=True)

        self.i_ogate = nn.Linear(input_size, hidden_size, bias=True)
        self.h_ogate = nn.Linear(hidden_size, hidden_size, bias=True)
        
        #init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    
    def forward(self, x, hidden):
        ht_prev, ct_prev = hidden

        fgate = self.sigmoid(self.i_fgate(x) + self.h_fgate(ht_prev))
        igate = self.sigmoid(self.i_igate(x) + self.h_igate(ht_prev))
        cellt_hat = self.tanh(self.i_cell(x) + self.h_cell(ht_prev))
        ogate = self.sigmoid(self.i_ogate(x) + self.h_ogate(ht_prev))

        ct_next = torch.mul(ct_prev, fgate) + torch.mul(igate, cellt_hat)
        ht_next = torch.mul(self.tanh(ct_next), ogate)

        return (ht_next, ct_next)
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = LSTMCell(input_size=input_size, hidden_size=hidden_size)

        self.fc = nn.Linear(hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        hidden = self.lstm_cell(x, hidden)
        h, _ = hidden
        out = self.fc(h)
        out = self.softmax(out)

        return out, hidden

    def initHidden(self):
        h0 = torch.zeros(1, self.hidden_size)
        c0 = torch.zeros(1, self.hidden_size)
        return h0, c0
```

<img src="img\image-20230610212306252.png" alt="image-20230610212306252" style="zoom:67%;" />

<img src="img\image-20230610212318603.png" alt="image-20230610212318603" style="zoom:67%;" />

<img src="img\image-20230610212403678.png" alt="image-20230610212403678" style="zoom:67%;" />

## GRU实现

自己实现的GRU代码：

```python
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.i_rgate = nn.Linear(input_size, hidden_size, bias=True)
        self.h_rgate = nn.Linear(hidden_size, hidden_size, bias=True)

        self.i_zgate = nn.Linear(input_size, hidden_size, bias=True)
        self.h_zgate = nn.Linear(hidden_size, hidden_size, bias=True)

        self.i_cell = nn.Linear(input_size, hidden_size, bias=True)
        self.h_cell = nn.Linear(hidden_size, hidden_size, bias=True)

        self.i_hgate = nn.Linear(input_size, hidden_size, bias=True)
        self.h_hgate = nn.Linear(hidden_size, hidden_size, bias=True)
        
        #init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    
    def forward(self, x, hidden):

        rgate = self.sigmoid(self.i_rgate(x) + self.h_rgate(hidden))
        zgate = self.sigmoid(self.i_zgate(x) + self.h_zgate(hidden))
        hgate = self.tanh(self.i_hgate(x) + self.h_hgate(torch.mul(hidden, rgate)))

        hidden_next = torch.mul(zgate, hidden) + torch.mul((1 - zgate), hgate)

        return hidden_next
    
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru_cell = GRUCell(input_size=input_size, hidden_size=hidden_size)

        self.fc = nn.Linear(hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        hidden = self.gru_cell(x, hidden)
        out = self.fc(hidden)
        out = self.softmax(out)

        return out, hidden

    def initHidden(self):
        h0 = torch.zeros(1, self.hidden_size).to(device)
        return h0
```

<img src="img\image-20230610232357844.png" alt="image-20230610232357844" style="zoom:67%;" />

<img src="img\image-20230610232427038.png" alt="image-20230610232427038" style="zoom:67%;" />

<img src="img\image-20230610232456113.png" alt="image-20230610232456113" style="zoom:67%;" />



## 分析

**为什么LSTM网络的性能优于RNN网络：**

1. 解决梯度消失问题：RNN存在梯度消失问题，即在反向传播过程中，梯度会随时间步长指数级地衰减，导致较早时间步的信息无法有效传递到后续时间步。LSTM通过引入门控机制，如遗忘门、输入门和输出门，可以有效地控制信息的流动，从而缓解了梯度消失问题。
2. 长期记忆能力：LSTM通过细胞状态（cell state）的概念来保持和传递长期记忆。细胞状态允许LSTM选择性地遗忘或存储信息，而不会受到门控机制的影响。这使得LSTM能够更好地处理长期依赖关系，例如在自然语言处理任务中理解句子的语义。
3. 处理序列中的长距离依赖：由于LSTM能够有效地保留长期记忆，它能够更好地处理序列中的长距离依赖。相比之下，传统的RNN对于长距离依赖的处理能力较弱，因为信息需要通过一系列的隐藏状态传递，而随着时间的增加，梯度的衰减会导致长距离的依赖信息丢失。

实验中德语名字预测效果不好，因为德语名字大部分都比较长，下面对比分析RNN、LSTM、GRU三种网络架构在预测Diefenbach这一德语名字：

```
RNN：
> Diefenbach
(0.37) Italian
(0.19) German
(0.09) Dutch

LSTM：
> Diefenbach
(1.00) German
(0.00) Russian
(0.00) Czech

GRU：
> Diefenbach
(0.74) German
(0.10) English
(0.10) Vietnamese
```

可见，LSTM和GRU都能成功预测出其时德语，因为这两种循环神经网络在处理长期依赖时的能力更强，而RNN对于这种长度很长的名字预测能力不好。