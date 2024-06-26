from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(SemGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))#我在想这是做了一个decouple吗，一个自身的，一个与邻居节点的
        nn.init.xavier_uniform_(self.W.data, gain=1.414)#保证输入输出方差一致，这里用的是均匀分布的xavier初始化

        self.adj = adj
        self.m = (self.adj > 0) #获得一个bool值矩阵
        self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))#e是一个包含长度为非零元素数量的向量
        nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])#k*token_size*tokn_size*next_token_size
        h1 = torch.matmul(input, self.W[1])

        adj = -9e15 * torch.ones_like(self.adj).to(input.device)#用很小的数来模拟0，
        adj[self.m] = self.e
        adj = F.softmax(adj, dim=1)

        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)#创建单位矩阵
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)#k*next_token_size
        #对于文章中的多通道思想，好像就是变化token_size的长度，即channel=token_size

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'