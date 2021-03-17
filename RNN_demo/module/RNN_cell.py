# @Time    : 2021/03/11 17:07
# @Author  : SY.M
# @FileName: RNN_cell.py


import torch

class RNN_cell(torch.nn.Module):
    def __init__(self,
                 d_input: int,
                 d_hidden: int,
                 number_of_classes: int,
                 batch_size: int):
        super(RNN_cell, self).__init__()

        self.rnn_cell = torch.nn.RNNCell(input_size=d_input, hidden_size=d_hidden)

        self.linear_out = torch.nn.Linear(in_features=d_hidden, out_features=number_of_classes)

        self.hidden = torch.zeros(batch_size, d_hidden)

    def forward(self, x):

        out = self.rnn_cell(x, self.hidden)

        self.hidden = out.data

        out = self.linear_out(out)

        return out