import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, space_dim, num_servers, hidden_dim, num_layers, bidirectional=True):
        super(LSTMModel, self).__init__()
        self.dim = space_dim
        self.num_servers = num_servers
        self.input_dim = (num_servers + 1) * space_dim
        self.hidden_dim = hidden_dim
        # First layer takes in n server location and the request location
        self.lstm = nn.LSTM( self.input_dim, hidden_dim, num_layers, bidirectional=bidirectional)
        # Should output the logits for which server to move
        self.linear = nn.Linear(hidden_dim, num_servers)

    def forward(self, req_sequence):
        lstm_out, _ = self.lstm(req_sequence.view(1, -1, self.input_dim))
        print(lstm_out.shape)
        logits = self.linear(lstm_out.view(1, -1, self.hidden_dim))
        print(logits.shape)
        server_scores = F.log_softmax(logits.view(self.num_servers, -1), dim=1)
        print(server_scores.shape)
        return server_scores
