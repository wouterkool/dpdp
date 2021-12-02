import math
import numpy as np
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.tsptw.state_tsptw import StateTSPTWInt
from utils.functions import accurate_cdist

class TSPTW(object):

    NAME = 'tsptw'  # TSP with Time Windows

    @staticmethod
    def get_costs(dataset, pi):
        """
        :param dataset: (batch_size, graph_size, 2) coordinates
        :param pi: (batch_size, graph_size) permutations representing tours
        :return: (batch_size) lengths of tours
        """
        # Check that tours are valid, i.e. contain 1 to n (0 is depot and should not be included)
        if (pi[:, 0] == 0).all():
            pi = pi[:, 1:]  # Strip of depot
        assert (
                torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) + 1 ==
                pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Distance must be provided in dataset since way of rounding can vary
        if 'dist' in dataset:
            dist = dataset['dist']
        else:
            coords = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
            dist = accurate_cdist(coords, coords).round().int()

        batch_size, graph_size, _ = dataset['loc'].size()

        # Check the time windows
        t = dist.new_zeros((batch_size, ))
        #assert (pi[:, 0] == 0).all()  # Tours must start at depot
        batch_zeros = pi.new_zeros((batch_size, ))
        cur = batch_zeros
        batch_ind = torch.arange(batch_size).long()
        lb, ub = torch.unbind(dataset['timew'], -1)
        for i in range(graph_size - 1):
            next = pi[:, i]
            t = torch.max(t + dist[batch_ind, cur, next], lb[batch_ind, next])
            assert (t <= ub[batch_ind, next]).all()
            cur = next

        length = dist[batch_ind, 0, pi[:, 0]] + dist[batch_ind[:, None], pi[:, :-1], pi[:, 1:]].sum(-1) + dist[batch_ind, pi[:, -1], 0]
        # We want to maximize total prize but code minimizes so return negative
        return length, None

    # @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPTWDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSPTWInt.initialize(*args, **kwargs)


def get_rounded_distance_matrix(coord):
    return cdist(coord, coord).round().astype(np.int)


def generate_instance(size):
    raise NotImplementedError()


class TSPTWDataset(Dataset):
    
    def __init__(self, filename=None, size=100, num_samples=1000000, offset=0, distribution=None, normalize=False):
        super(TSPTWDataset, self).__init__()

        self.data_set = []
        assert filename is not None
        assert not normalize
        assert os.path.splitext(filename)[1] == '.pkl'

        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.data = [
                {
                    'loc': torch.tensor(loc, dtype=torch.float),
                    'depot': torch.tensor(depot, dtype=torch.float),
                    'timew': torch.tensor(timew, dtype=torch.int64),
                    'max_coord': torch.tensor(max_coord, dtype=torch.int64),  # Scalar
                }
                for depot, loc, timew, max_coord in (data[offset:offset+num_samples])
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

