# RNN_demo
基础RNN的使用，通过小练习帮助理解RNN中的维度

## 小练习描述
基于pytorch框架，分别使用基础代码、RNNcell和RNN网络构建神经网络。目标是将字符串'hello',通过学习变成'ohlol'。

## 环境
环境|版本|
----|----|
框架|pytorch1.6|
IDE|pycharm|
操作系统|WINDOWS10|
设备|CPU|

## 训练数据
使用两个字符串 源字符串——'hello'  目标字符串——'ohlol'，尝试通过RNN网络将输入的源字符串变成目标字符串。<br>
字母数字键值对为：
```
    {'o': 0, 0: 'o', 'e': 1, 1: 'e', 'l': 2, 2: 'l', 'h': 3, 3: 'h'}
```
源数据向量按照one-hot进行编码，dataset_x:<br>
```
    tensor([[0., 0., 0., 1.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.]])
```
目标数据根据键值对进行编码，dataset_y:<br>
```
    tensor([0., 3., 2., 0., 2.])
```

## 模型及维度描述

![Image text](https://github.com/SY-Ma/RNN_demo/blob/main/image/RNNcell%E7%BD%91%E7%BB%9C%E6%9E%B6%E6%9E%84%E5%9B%BE.png)<br>

由于仅有一条数据，我们不再使用DataLoader和Dataset进行批量处理，因此我们直接将每一个时间步的数据进行输入。各变量的含义描述如下：<br>

变量|描述|
----|-----|
d_input| 每个时间步向量的维度|
d_hidden| 经过Linear后hidden的维度|
BATCH_SIZE|由于仅有一条数据，所以值为1|
number_of_classes|一共是4个字母，所以最终输出向量的维度是4维|
num_layers|RNN层数|
seq_len|时间序列的长度|

### RNN_basic_code

RNNcell有两个输入，一个是上一层的hidden输出，一个是时间t的输入，两个数据进入RNNcell之后其实就是经过了两个线性层改变了维度，然后加在一起经过tanh激活函数,就得到了t时刻的hidden输出。RNN_basic_code不使用Pytorch提供的torch.nn.RNNCell,而是自己构建线性层和激活函数，对数据进行处理，每次循环，我们都将输出的hidden进行一次线性变换映射到输出维度进行loss的计算。<br>

各阶段数据维度如下：<br>

数据|shape|
----|-----|
X_t|(d_input)|
Hidden_t|(d_hidden)|
Hidden_t+1|(d_hidden)|
计算损失时的预测数据维度|(1, number_of_classes)|
计算损失时的目标数据维度|(1)|

看到X_t的维度我们可以知道，完成循环我们需要使用for循环每次取出一个时间步的向量，且在一次循环中计算loss和反向传播。

### RNNCell
使用torch.nn.RNNCell直接创建RNNcell,同样的在RNNcell之后接上线性层和交叉熵损失函数。与RNN_basic_code不同的是，对于输入的维度，我们需要添加上一个batch_size变量，因为torch框架都是分批处理数据的，我们可以一次循环多个样本的数据。<br>
各阶段数据维度如下：<br>

数据|shape|
----|-----|
X_t|(batch_size, d_input)|
Hidden_t|(batch_size, d_hidden)|
Hidden_t+1|(batch_size, d_hidden)|
计算损失时的预测数据维度|(batch_size, number_of_classes)|
计算损失时的目标数据维度|(1)|

其实RNNCell和RNN_basic_code的维度含义是相同的，RNN——basic_code的输入变量维度其实也可以是(batch_size, d_input)，我们只不过是在实现的时候结合数据仅有一条的情况简化了以下，其在计算损失之前进行了unsqueeze(dim=0)的操作，其实就是加了个batch_size=1的维度。

### RNN
使用torch.nn.RNN直接创建RNN模型，与RNNCell不同的是，RNN加了一个维度。我们不在需要for循环取每个时间步的向量，而是直接将batch_size个样本(这里batch_size为1)全部输入。同样的，labels为batch_size个样本的所有时间步向量。<br>
各阶段数据维度如下：<br>

数据|shape|
----|-----|
X_t|(batch_size, seq_len, d_input)|
Hidden_t|(batch_size, num_layers, d_hidden)|
Hidden_t+1|(batch_size, num_layers, d_hidden)|
计算损失时的预测数据维度|(batch_size * seq_len, number_of_classes)|
计算损失时的目标数据维度|(batch_size * seq_len)|

RNN会返回两个值，一个是每次RNNcell循环得到的hidden组成的列表，一个是最后一次RNNcell循环得到的hidden。维度分别为(batch_size, seq_len, d_hidden)  (num_layers, batch_size, d_hidden)<br>
较难理解的是计算损失时的维度情况，网络输出的每个时间步的预测值维度为(batch_size, seq_len, number_of_classes)，目标向量的维度为(batch_size, seq_len)，而交叉熵损失函数要求预测向量维度为(N, C)，目标向量维度为(N)，其中C为number_of_classes，所以对于目标向量，我们对其进行降维，变为(batch_size * seq_len)，相应的预测向量为维度应该满足损失函数的要求，变为(batch_size * seq_len, number_of_classes)，相当于将每个样本的时间序列展开来，结合下图更容易理解。

![Image text](https://github.com/SY-Ma/RNN_demo/blob/main/image/RNN%E8%AE%A1%E7%AE%97%E6%8D%9F%E5%A4%B1%E5%90%91%E9%87%8F%E7%BB%B4%E5%BA%A6.png)<br>

## 练习结果
每个模型基本上都能在5个Epoch左右变成'ohlol',改变目标字符串也能达到同样的速度。
```
Epoch:1		loss:1.2994171380996704		Predict:ooooh		Target:ohlol
Epoch:2		loss:0.979849636554718		Predict:oolol		Target:ohlol
Epoch:3		loss:0.7345105409622192		Predict:ollol		Target:ohlol
Epoch:4		loss:0.4432271122932434		Predict:ohlol		Target:ohlol
Epoch:5		loss:0.2078162431716919		Predict:ohlol		Target:ohlol
```

## 参考
基础知识视频：https://www.bilibili.com/video/BV1Y7411d7Ys?p=12


## 本人学识浅薄，代码和文字若有不当之处欢迎批评与指正！
## 联系方式：masiyuan007@qq.com
