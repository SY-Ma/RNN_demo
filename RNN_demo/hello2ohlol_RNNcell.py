# @Time    : 2021/03/09 17:36
# @Author  : SY.M
# @FileName: hello2ohlol_RNNcell.py

import torch
from module.RNN_cell import RNN_cell

original = 'hello'
target = 'ohlol'

# original = original.strip()
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
BATCH_SZIE = 1
number_of_classes = 4

net = RNN_cell(d_input=d_input, d_hidden=d_hidden, number_of_classes=number_of_classes, batch_size=BATCH_SZIE)
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
loss_function = torch.nn.CrossEntropyLoss()

for step in range(EPOCH):
    predict = ''
    loss = 0.0
    for index, time_step in enumerate(dataset_x):
        optimizer.zero_grad()
        out = net(time_step.unsqueeze(dim=0))  # input_size (batch_size, d_input)
        loss_temp = loss_function(out, dataset_y[index: index+1].long())
        loss += loss_temp
        loss_temp.backward()
        optimizer.step()

        out = out.squeeze().detach().numpy().tolist()
        predict = predict + letter_dict[out.index(max(out))]

    print(f'Epoch:{step+1}\t\tloss:{loss.item()}', end='\t\t')
    print(f'Predict:{predict}\t\tTarget:{target}')
