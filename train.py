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

use_cuda = torch.cuda.is_available()
if use_cuda:
    if parser.manual_seed >= 0:
    	torch.cuda.manual_seed(parser.manual_seed)
device = torch.device("cuda" if use_cuda else "cpu")

def train(epoch, model, training_set, optimizer, writer, verbose=True):
    total_optimal_cost = 0
    total_model_cost = 0
    total_model_loss = 0
    model.train()
    for i, (X_batch, y_batch) in enumerate(tqdm.tqdm(training_set)):
        # first, get the list of servers
        servers = training_set.datasets[i].servers
        optimal_cost = training_set.datasets[i].cost
        model_cost = 0
        locations = torch.Tensor(servers, device=device)
        total_loss = 0
        optimizer.zero_grad()
        for X, y in zip(X_batch, y_batch):
            # go through each example, get the starting points, etc.
            X_all = torch.cat(locations, X)
            # log_probs is the log probability of the elements
            log_probs = model(X_all)
            model_loss = F.nll_loss(log_probs, y)
            total_loss += model_loss
            # Gives the index of the server to move
            # TODO: See if we should change it to a probabilistic model
            model_pred = log_probs.argmax()
            model_cost += distance_function(locations[i], y).item()
            locations[i] = y
        total_model_loss += (model_loss.item())
        model_loss.backwards()
        optimizer.step()
    if verbose:
        print('Epoch {}/{}: \t\tModel cost/Optimal Cost: {}/{}\n\t\t\tRatio: {} Loss: {}'.format(
            i, len(training_set), total_model_cost, total_optimal_cost, 
            total_model_cost/total_optimal_cost, total_model_loss))


def test(epoch, model, test_set, writer, verbose=True):
    total_optimal_cost = 0
    total_model_cost = 0
    total_model_loss = 0
    model.test()
    for i, (X_batch, y_batch) in enumerate(tqdm.tqdm(test_set)):
        # first, get the list of servers
        servers = test_set.datasets[i].servers
        optimal_cost = test_set.datasets[i].cost
        model_cost = 0
        locations = torch.Tensor(servers, device=device)
        total_loss = 0
        for X, y in zip(X_batch, y_batch):
            # go through each example, get the starting points, etc.
            X_all = torch.cat(locations, X)
            # log_probs is the log probability of the elements
            log_probs = model(X_all)
            model_loss = F.nll_loss(log_probs, y)
            total_loss += model_loss
            # Gives the index of the server to move
            model_pred = log_probs.argmax()
            model_cost += distance_function(locations[i], y).item()
            locations[i] = y
        total_model_loss += (model_loss.item())
    if verbose:
        print('Testing:')
        print('Epoch {}/{}: \t\tModel cost/Optimal Cost: {}/{}\n\t\t\tRatio: {} Loss: {}'.format(
            i, len(training_set), total_model_cost, total_optimal_cost, 
            total_model_cost/total_optimal_cost, total_model_loss))

def save_model(epoch, model):
    pass

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

def distance_function(x, y, metric=metric):
    return torch.norm(x-y, p=metric)

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
model = model.to(device)
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
                                                dimensions=dims, distance_metric=metric, device=device)

if __name__ == '__main__':
    for e in range(epochs):
        train(e, model, training_set, optimizer, writer)
        test(e, model, test_set, writer)