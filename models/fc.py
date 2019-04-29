import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import numpy as np


class FCModel(nn.Module):
    def __init__(self, space_dim, num_servers, hidden_dim, num_hidden_layers):
        super(FCModel, self).__init__()
        self.dim = space_dim
        self.num_servers = num_servers
        self.input_dim = (num_servers + 1) * space_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_hidden_layers
        layers = self._build_layers()
        self.model = nn.Sequential(*layers)

    def _build_layers(self):
        self.first_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.layers = [self.first_layer, nn.ReLU()]# , nn.BatchNorm1d(self.hidden_dim)]
        for i in range(self.num_layers):
            self.layers += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()]#, nn.BatchNorm1d(self.hidden_dim)]
        self.final_layer = nn.Linear(self.hidden_dim, self.num_servers)
        self.layers.append(self.final_layer)
        return self.layers

    def forward(self, req_sequence):
        # print("We here {}".format(req_sequence.view(-1, self.input_dim).shape))
        input = req_sequence.view(-1, self.input_dim)
        for layer in self.layers:
            input = layer(input)
        logits = input
        server_scores = F.log_softmax(logits, dim=1)
        return server_scores
