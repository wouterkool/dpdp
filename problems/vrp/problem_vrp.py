from torch.utils.data import Dataset
import torch
import os
import pickle

from .generate_uchoa_data import generate_uchoa_instances


class CVRP(object):

    NAME = 'cvrp'  # Capacitated Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi, allow_infeas=False):
        batch_size, graph_size = dataset['demand'].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # vehicle_capacity = dataset.get('capacity', torch.full_like(dataset['demand'][:, :1], CVRP.VEHICLE_CAPACITY))
        vehicle_capacity = dataset.get('capacity', dataset['demand'].new_full((batch_size, ), CVRP.VEHICLE_CAPACITY)).to(dataset['demand'].dtype)
        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                -vehicle_capacity.view(batch_size, 1),
                dataset['demand']
            ),
            1
        )
        d = demand_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        for i in range(pi.size(1)):
            used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert allow_infeas or (used_cap <= vehicle_capacity + 1e-5).all(), "Used more than capacity"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)


def make_instance(args, normalize=True):
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    if normalize:
        return {
            'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
            'demand': torch.tensor(demand, dtype=torch.float) / capacity,
            'depot': torch.tensor(depot, dtype=torch.float) / grid_size
        }
    return {  # Let dtypes be inferred
        'loc': torch.tensor(loc),
        'demand': torch.tensor(demand),
        'depot': torch.tensor(depot),
        'capacity': capacity,
        'grid_size': grid_size
    }


class VRPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None, normalize=True):
        super(VRPDataset, self).__init__()

        if distribution is None:
            distribution = "nazari"

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args, normalize) for args in data[offset:offset+num_samples]]

        else:
            if distribution == "nazari":
                # From VRP with RL paper https://arxiv.org/abs/1802.04240
                CAPACITIES = {
                    10: 20.,
                    20: 30.,
                    50: 40.,
                    100: 50.,
                }

                self.data = [
                    {
                        'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                        # Uniform 1 - 9, scaled by capacities
                        'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                        'depot': torch.FloatTensor(2).uniform_(0, 1)
                    } if normalize else
                    {
                        'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                        # Uniform 1 - 9, scaled by capacities
                        'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1),
                        'depot': torch.FloatTensor(2).uniform_(0, 1),
                        'capacity': CAPACITIES[size]
                    }
                    for i in range(num_samples)
                ]
            else:
                # Uchoa et al. New_Benchmark_Instances_for_the_Capacitated_Vehicle_Routing_Problem
                self.data = [
                    {
                        'loc': loc / grid_size.float(),
                        'demand': demand.float() / capacity.float(),
                        'depot': depot / grid_size.float()
                    } if normalize else
                    {
                        'loc': loc,
                        'demand': demand,
                        'depot': depot,
                        'capacity': capacity,
                        'grid_size': grid_size
                    }
                    for depot, loc, demand, capacity, depot_types, customer_types, grid_size
                    in zip(*generate_uchoa_instances(num_samples, size, distribution))
                ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
