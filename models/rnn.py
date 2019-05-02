import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, space_dim, num_servers, 
                 hidden_dim, num_layers, bidirectional=False, 
                 training_seq_size=100, test_seq_size=1):
        super(LSTMModel, self).__init__()
        self.dim = space_dim
        self.num_servers = num_servers
        self.input_dim = (num_servers + 1) * space_dim
        self.hidden_dim = hidden_dim
        # First layer takes in n server location and the request location
        self.lstm = nn.LSTM( self.input_dim, hidden_dim, num_layers, bidirectional=bidirectional)
        # Should output the logits for which server to move
        self.linear = nn.Linear(hidden_dim, num_servers)
        self.training_seq_size = training_seq_size
        self.test_seq_size = test_seq_size

    def train(self, is_train=True):
        self.is_train = is_train
        super(LSTMModel, self).train(is_train)
    
    def eval(self):
        self.is_train = False
        super(LSTMModel, self).eval()

    def forward(self, req_sequence):
        seq_length = self.training_seq_size if self.is_train else self.test_seq_size
        lstm_out, _ = self.lstm(req_sequence.view(seq_length, -1, self.input_dim))
        logits = self.linear(lstm_out.view(seq_length, self.hidden_dim))
        server_scores = F.log_softmax(logits.view(-1, self.num_servers), dim=1)
        return server_scores
