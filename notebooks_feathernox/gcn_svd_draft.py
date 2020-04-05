import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch import nn
import torch.nn.functional as F


class GraphConvolutionBSSVD(Module):
    def __init__(self, orig_layer, r=None):
        super(GraphConvolutionBSSVD, self).__init__()
        
        
        self.in_features = orig_layer.in_features
        self.out_features = orig_layer.out_features
        self.sigma = orig_layer.sigma
        
        if r is None:
            r = self.get_max_rank()
        
        self.r = r
        
        U, s, Vt = torch.svd(orig_layer.weight.clone())
        Vt = Vt.T
        self.A = Parameter((U[:, :self.r] * s[:self.r]) @ Vt[:self.r, :])
        
        if self.training:
            self.U, self.s, self.Vt = None, None, None
        else:
            self.U, self.s, self.Vt = torch.svd(self.A)
        
        if orig_layer.bias is not None:
            self.bias = Parameter(orig_layer.bias.clone())
        else:
            self.register_parameter('bias', None)


    def forward(self, input, adj):
        U, s, Vt = self.svd()
        
        support = torch.mm(input, U[:, :self.r])
        support = support * s[:self.r]
        support = torch.mm(support, Vt[:self.r, :])
        output = torch.spmm(adj, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return self.sigma(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) \
               + ', ' + str(self.r) + ')'
    
    def set_rank(self, r=None):
        if r is None:
            self.r = self.get_max_rank()
        else:
            self.r = r
        
    def get_max_rank(self):
        return min(self.in_features, self.out_features)
    
    def svd(self):
        if self.training:
            self.U, self.s, self.Vt = None, None, None
            U, s, Vt = torch.svd(self.A)
            Vt = Vt.T
            return U, s, Vt
        
        if self.U is None:
            self.U, self.s, self.Vt = torch.svd(self.A)
            self.Vt = self.Vt.T
        
        return self.U, self.s, self.Vt
