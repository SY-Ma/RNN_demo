# RNN_demo
基础RNN的使用，通过小练习帮助理解RNN中的维度

## 小练习描述
基于pytorch框架，分别使用基础代码、RNNcell和RNN网络构建神经网络。目标是将字符串'hello',通过学习变成'ohlol'。

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
### RNN_basic_code
![Image text](https://github.com/SY-Ma/RNN_demo/blob/main/image/RNNcell%E7%BD%91%E7%BB%9C%E6%9E%B6%E6%9E%84%E5%9B%BE.png)
RNNcell有两个输入，一个是上一层的hidden输出，一个是时间t的输入，两个数据进入RNNcell之后其实就是经过了两个线性层改变了维度，然后加在一起经过tanh激活函数,就得到了t时刻的hidden输出。RNN_basic_code不使用Pytorch提供的torch.nn.RNNCell,而是自己构建线性层和激活函数，对数据进行处理。
