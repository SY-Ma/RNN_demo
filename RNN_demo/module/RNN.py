# @Time    : 2021/03/11 17:31
# @Author  : SY.M
# @FileName: RNN.py


import torch

class RNN(torch.nn.Module):
    def __init__(self,
                 d_input: int,
                 d_hidden: int,
                 num_layers: int,
                 number_of_classes: int,
                 batch_size: int):
        super(RNN, self).__init__()

        self.rnn = torch.nn.RNN(input_size=d_input, hidden_size=d_hidden, num_layers=num_layers, batch_first=True)

        self.linear_out = torch.nn.Linear(in_features=d_hidden, out_features=number_of_classes)

        self.init_hidden = torch.zeros(batch_size, num_layers, d_hidden)

    def forward(self, x):

        out, _ = self.rnn(x, self.init_hidden)

        out = self.linear_out(out)

        return out



