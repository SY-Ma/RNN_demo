# @Time    : 2021/03/19 16:56
# @Author  : SY.M
# @FileName: hello2ohlol_basic_code.py

import torch
from module.RNN_basic_code import RNN_basic_code

original = 'hello'
target = 'ohlol'

letters = set(original)

letter_dict = {}
for index, letter in enumerate(letters):
    letter_dict[letter] = index
    letter_dict[index] = letter

print(letter_dict)

one_hot_table = [[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]

x = []
for i in original:
    x.append(one_hot_table[letter_dict[i]])
y = []
for i in target:
    y.append(letter_dict[i])
dataset_x = torch.Tensor(x)
dataset_y = torch.Tensor(y)
print('dataset_x:\r\n ', dataset_x)
print('dataset_y:\r\n ', dataset_y)

EPOCH = 15
d_hidden = 8
d_input = 4
BATCH_SIZE = 1
number_of_class = 4

init_hidden = torch.zeros(d_hidden)
net = RNN_basic_code(d_input=d_input, d_hidden=d_hidden, number_of_class=4)
optimizer = torch.optim.Adam([{'params': net.parameters()}], lr=0.1)
loss_function = torch.nn.CrossEntropyLoss()

for step in range(EPOCH):
    predict = ''
    loss = 0
    for index, time_step in enumerate(dataset_x):
        optimizer.zero_grad()

        out, hidden = net(time_step, init_hidden)  # input_size (d_input)
        init_hidden = hidden.data  # 注意加.data 因为hidden是一个 允许梯度下降的量，不能将其引用传给其他变量
        loss_temp = loss_function(out.unsqueeze(dim=0), dataset_y[index:index+1].long())
        loss_temp.backward()
        loss = loss + loss_temp
        optimizer.step()

        out = out.squeeze().detach().numpy().tolist()
        predict = predict + letter_dict[out.index(max(out))]


    print(f'Epoch:{step+1}\t\tloss:{loss.item()}', end='\t\t')
    print(f'Predict:{predict}\t\tTarget:{target}')

