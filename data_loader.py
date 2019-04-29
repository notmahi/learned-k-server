import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import tqdm

from optimal_offline import KServer


class AbstractDistribution(object):
    """
    Abstract class for all distrubitons.
    """
    def __init__(self):
        raise NotImplementedError

    def sample(self, n):
        raise NotImplementedError


class NumpyDistribution(AbstractDistribution):
    """
    Reference wrapper around numpy distributions.
    """
    def __init__(self, distribution_name='normal', shift= np.array([0]), args=(1.,), seed=0):
        self.distribution = getattr(np.random, distribution_name)
        self.args = args
        self.shift = shift
        np.random.seed(seed)

    def sample(self, n):
        sz = (n,) + np.array(self.shift).shape
        return self.shift + self.distribution(*self.args, size=sz)


class MixedDistribution(AbstractDistribution):
    """
    Reference implementation of mixture models. Takes a list of distributions and 
    their associated, and then allows you to sample from the mixed models.
    """
    def __init__(self, distributions, weights, seed=0):
        assert len(distributions) == len(weights)
        self.distributions = distributions
        self.weights = np.array(weights, dtype='f')
        self.weights /= self.weights.sum()
        np.random.seed(seed)

    def sample(self, n):
        num_samples = np.rint([n * p for p in self.weights]).astype(int)
        samples = [self.distributions[i].sample(num_sample) for i, num_sample in enumerate(num_samples)]
        if np.sum(num_samples) < n:
            choices = np.random.choice(len(self.distributions), (n - np.sum(num_samples)), p=self.weights)
            for c in choices:
                samples.append(self.distributions[c].sample(1))
        samples = np.concatenate(samples)
        np.random.shuffle(samples)
        return samples

class KServerDataset(Dataset):
    """
    A generator for k-server datasets.
    """
    def __init__(self, num_servers, 
                 num_requests, server_distribution, request_distribution, 
                 dimensions=2, distance_metric=2, seed=0, device='cpu'):
        np.random.seed(seed)
        self.servers = server_distribution.sample(num_servers)
        self.requests = request_distribution.sample(num_requests)
        self.instance = KServer(servers = self.servers, requests= self.requests, order=distance_metric)
        self.cost = self.instance.optimal_cost()
        self.optimal_movement = self.instance.get_serves()

        if type(device) == str:
            device = torch.device(device)
        self.servers = torch.Tensor(self.servers, device=device)
        self.requests = torch.Tensor(self.requests, device=device)
        self.optimal_movement = torch.Tensor(self.optimal_movement, device=device).type(torch.LongTensor)
        self.optimal_movement.unsqueeze_(-1)

    def __len__(self):
        return len(self.optimal_movement)
    
    def __getitem__(self, idx):
        return (self.requests[idx], self.optimal_movement[idx])


class ConstantDistribution(AbstractDistribution):
    def __init__(self, points):
        self.points = points
    
    def sample(self, n):
        assert len(self.points) == n
        return self.points


def _kserver_training_set(len_data, num_servers, 
            num_requests, server_distribution, request_distribution, 
            dimensions=2, distance_metric=2, seed=0, device='cpu'):
    np.random.seed(seed)
    batch_size = num_requests
    single_datasets = []
    for i in tqdm.trange(len_data // num_requests):
        single_datasets.append(KServerDataset(num_servers, 
                                num_requests, server_distribution, 
                                request_distribution, dimensions, 
                                distance_metric, seed = np.random.randint(0, len_data), 
                                device=device))
    return ConcatDataset(single_datasets)


def _kserver_loader(len_data, num_servers, 
            num_requests, server_distribution, request_distribution, 
            dimensions=2, distance_metric=2, seed=0, device='cpu'):
    dataset = _kserver_training_set(len_data, num_servers, 
                                   num_requests, server_distribution, 
                                   request_distribution, dimensions, 
                                   distance_metric, seed, device=device)
    return DataLoader(dataset, batch_size=num_requests, shuffle=False, num_workers=2)


def kserver_test_and_train(len_train, len_test, num_servers, num_requests,
                           server_distribution, request_distribution, 
                           dimensions=2, distance_metric=2, seed=0, device='cpu'):
    train_loader = _kserver_loader(len_train, num_servers, num_requests, server_distribution, 
                                      request_distribution, dimensions, distance_metric, seed, device=device)
    test_loader = _kserver_loader(len_test, num_servers, num_requests, server_distribution, 
                                      request_distribution, dimensions, distance_metric, seed, device=device)
    return train_loader, test_loader


def distribution_from_centers(mus, sigmas, weights=None, seed=0):
    dist = []
    if weights is None:
        weights = [1.] * len(mus)
    for mu, sigma in zip(mus, sigmas):
        dist.append(NumpyDistribution(distribution_name='normal', shift = mu, args=(sigma,), seed=seed))
    return MixedDistribution(dist, weights)


"""
Testing the code.

d = distribution_from_centers(np.array([[1,1],
                                        [-1,1],
                                        [-1,-1],
                                        [1,-1],
                                        [1,0],
                                        [0,1],
                                        [-1,0],
                                        [0,-1],
                                        [0,0]
                                        ], dtype='f'), np.array([0.5] * 9))

print(d.sample(100))
"""