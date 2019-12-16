import torch
import torch.nn as nn
import math


class Projections(nn.Module):

    def __init__(self, n_heads, embed_dim):
        super().__init__()

        self.n_heads = n_heads
        val_dim = embed_dim // n_heads

        self.W_key = nn.Parameter(torch.Tensor(n_heads, embed_dim, val_dim), requires_grad=True)
        self.W_val = nn.Parameter(torch.Tensor(n_heads, embed_dim, val_dim), requires_grad=True)
        self.W_output = nn.Parameter(torch.Tensor(1, embed_dim, embed_dim), requires_grad=True)


        self.init_parameters()


    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)


    def forward(self, h):

        batch_size, graph_size, input_dim = h.size()
        hflat = h.contiguous().view(-1, input_dim)

        shp = (self.n_heads, batch_size, graph_size, -1)

        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)
        V_output = torch.bmm(h, self.W_output.repeat(batch_size, 1, 1))

        output = {
            'K' : K,
            'V': V,
            'V_output': V_output
        }

        return output
