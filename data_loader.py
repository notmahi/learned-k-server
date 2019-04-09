import numpy as np

from torch.utils.data import Dataset

from optimal_offline import KServer

class Distribution(object):
    """
    Reference wrapper around numpy distributions.
    """
    def __init__(self, distribution_name='normal', args=(0., 1.), seed=0):
        self.distribution = getattr(np.random, distribution_name)
        np.random.seed(seed)

    def sample(self, n):
        return self.distribution(*self.args, n)

class MixtureModels(object):
    """
    Reference implementation of mixture models. Takes a list of distributions and 
    their associated, and then allows you to sample from the mixed models.
    """
    def __init__(self, distributions, weights, seed=0):
        assert len(distributions) == len(weights)
        self.distributions = distributions
        self.weights = np.array(weights, dtype='f')
        self.weights /= weights.sum()
        np.random.seed(seed)

    def sample(self, n):
        num_samples = np.rint([n * p for p in self.weights])
        samples = [self.distributions.sample(num_sample) for num_sample in num_samples]
        if np.sum(num_samples) < n:
            choices = np.random.choice(len(self.distributions), (n - np.sum(num_samples), p=self.weights))
            for c in choices:
                samples.append(self.distributions[c].sample(1))
        samples = np.concatenate(samples)
        np.random.shuffle(samples)
        return samples

class KServerDataset(Dataset):
    I"""
    A generator for k-server datasets.
    """
    def __init__(self, num_servers, 
                 num_requests, server_distribution, request_distribution, 
                 dimensions=2, distance_metric=2, seed=0):
        np.random.seed(seed)
        self.servers = server_distribution.sample(num_servers)
        self.requests = request_distribution.sample(num_requests)
        self.instance = KServer(servers = self.servers, requests= self.requests, order=distance_metric)
        self.cost = self.instance.optimal_cost()
        self.optimal_movement = self.instance.get_serves()

    def __len__(self):
        return len(self.optimal_movement)
    
    def __getitem__(self, idx):
        return (self.requests[idx], self.optimal_movement[idx])



