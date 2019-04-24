import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import json
import os
import numpy as np
from tensorboardX import SummaryWriter
import tqdm

from data_loader import *
from models import model_dict
from util import get_parser

def train(epoch, model, training_set, optimizer, writer, verbose=True):
    pass

def test(epoch, model, test_set, writer, verbose=True):
    pass

def save_model(epoch, model):
    pass

use_cuda = torch.cuda.is_available()
if use_cuda:
    if parser.manual_seed >= 0:
    	torch.cuda.manual_seed(parser.manual_seed)
device = torch.device("cuda" if use_cuda else "cpu")

parser = get_parser()
parser.parse_args()

# Summary writer and model directory
model_dir = parser.model_dir
# Initialize tensorboard writer
writer = SummaryWriter(model_dir)

# Training/test set details
num_servers = parser.n_servers
training_set_size = parser.n_requests_train
batch_size = parser.batch_size
test_set_size = parser.n_requests_test
dims = parser.dims
metric = parser.dist_metric

# Model params
architecture = parser.model
hidden_layers = parser.hidden_layers
hidden_units = parser.hidden_units

# Training params
learning_rate = parser.learning_rate
epochs = parser.epochs
optimizer = optim.SGD if parser.optim == 'sgd' else optim.Adam

# Create model
model = model_dict.get(architecture)
if model == None:
    model = model_dict['lstm']

model = model(dims, num_servers, hidden_units, hidden_layers)
optimizer = optimizer(model.parameters(), lr=learning_rate)

# Load dataset
# TODO (Mahi): Get a better way of inputting the distributions
server_distribution = distribution_from_centers([np.array([0., 0.])], [np.array([0.])])
request_distribution = distribution_from_centers(np.array([[ 1, 1],
                                                            [-1, 1],
                                                            [-1,-1],
                                                            [ 1,-1],
                                                            [ 1, 0],
                                                            [ 0, 1],
                                                            [-1, 0],
                                                            [ 0,-1],
                                                            [ 0, 0]], dtype='f'), np.array([0.5] * 9))

training_set, test_set = kserver_test_and_train(training_set_size, 
                                                test_set_size, 
                                                num_servers, batch_size, 
                                                server_distribution, request_distribution, 
                                                dimensions=dims, distance_metric=metric)

if __name__ == '__main__':
    for e in range(epochs):
        train(e, model, training_set, optimizer, writer)
        test(e, model, test_set, writer)