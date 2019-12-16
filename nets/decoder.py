import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np



class ClassifierOutput(nn.Module):

    def __init__(self, embedding_size, C=10, softmax_output=False):
        super().__init__()

        self.C = C
        self.embedding_size = embedding_size
        self.softmax_output = softmax_output

        self.W_q = nn.Parameter(torch.Tensor(1, embedding_size, embedding_size))

        self.init_parameters()


    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)


    def forward(self, context, V_output):

        self.batch_size = context.shape[0]

        Q = torch.bmm(context, self.W_q.repeat(self.batch_size, 1, 1))
        z = torch.bmm(Q, V_output.permute(0, 2, 1))
        z = z / (self.embedding_size ** (0.5))
        z = self.C * torch.tanh(z)

        if self.softmax_output:
            output = F.softmax(z, dim=1)
        else:
            output = z

        return output



class Attention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super().__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))


        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()


    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)


    def forward(self, q, K, V, mask=None):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """

        _, batch_size, graph_size, _ = K.size()
        _, n_query, input_dim = q.size()

        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)


        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = F.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out



class Decoder(nn.Module):

    def __init__(self, num_heads, input_size, embedding_size, softmax_output=False, C=10):
        super().__init__()


        self.C = C
        self.embedding_size = embedding_size

        self.initial_embedding = nn.Linear(in_features=input_size, out_features=embedding_size)
        self.attention = Attention(n_heads=num_heads, embed_dim=embedding_size, input_dim=embedding_size)

        self.classifier_output = ClassifierOutput(embedding_size=self.embedding_size, C=C, softmax_output=softmax_output)


    def forward(self, decoder_input, projections, mask, *args, **kwargs):

        mask = (mask == 0)

        K = projections['K']
        V = projections['V']
        V_output = projections['V_output']

        embedded_input = self.initial_embedding(decoder_input)
        context = self.attention(embedded_input, K, V, mask)
        output = self.classifier_output(context, V_output)
        return output
