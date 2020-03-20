import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class GraphConvolutionBSSVD(nn.Module):
    def __init__(self, orig_layer, r):
        super(GraphConvolutionBSSVD, self).__init__()
        self.r = r
        
        self.in_features = orig_layer.in_features
        self.out_features = orig_layer.out_features
        self.sigma = orig_layer.sigma
        
        U, s, V = torch.svd(orig_layer.weight)
        
        self.U = nn.Parameter(U[:, :r])
        self.s = nn.Parameter(s[:r])
        self.Vt = nn.Parameter(V.T[:r, :])
        
        if orig_layer.bias is not None:
            self.bias = Parameter(orig_layer.bias)
        else:
            self.register_parameter('bias', None)


    def forward(self, input, adj):
        support = torch.mm(input, self.U)
        support = support * self.s
        support = torch.mm(support, self.Vt)
        output = torch.spmm(adj, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return self.sigma(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) \
               + ', ' + str(self.r) + ')'