# @Time    : 2021/03/20 16:06
# @Author  : SY.M
# @FileName: RNN_basic_code.py

import torch

class RNN_basic_code(torch.nn.Module):
    def __init__(self,
                 d_input: int,
                 d_hidden: int,
                 number_of_class: int):
        '''

        :param d_input: 时间序列的某一个时间步的向量维度
        :param d_hidden: hidden向量维度
        '''
        super(RNN_basic_code, self).__init__()
        self.linear_for_x = torch.nn.Linear(in_features=d_input, out_features=d_hidden)
        self.linear_for_hidden = torch.nn.Linear(in_features=d_hidden, out_features=d_hidden)

        self.tanh = torch.nn.Tanh()

        self.linear_out = torch.nn.Linear(d_hidden, number_of_class)

    def forward(self,
                dataset_x: torch.Tensor,
                init_hidden:  torch.Tensor):

        x_temp = self.linear_for_x(dataset_x)

        hidden_temp = self.linear_for_hidden(init_hidden)

        out = self.tanh(x_temp + hidden_temp)

        hidden = out

        out = self.linear_out(out)

        return out, hidden