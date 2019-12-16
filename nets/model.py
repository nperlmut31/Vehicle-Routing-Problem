import torch
import torch.nn as nn
import torch.nn.functional as F
from fleet_beam_search_2.nets.decoder_2 import Decoder
from fleet_beam_search_2.nets.projections import Projections
from fleet_beam_search_2.nets.self_attention import Encoder


class Model(nn.Module):

    def __init__(self, input_size, decoder_input_size, embedding_size,
                 num_heads=8, num_layers=4, ff_hidden=250, *args, **kwargs):
        super().__init__()

        self.embedding_size = embedding_size

        self.encoder = Encoder(n_heads=num_heads,
                               embed_dim=embedding_size,
                               n_layers=num_layers,
                               feed_forward_hidden=ff_hidden,
                               node_dim=input_size)

        self.decoder = Decoder(input_size=4*embedding_size + decoder_input_size,
                               embedding_size=embedding_size,
                               num_heads=num_heads)

        self.projections = Projections(n_heads=num_heads,
                                       embed_dim=embedding_size)



