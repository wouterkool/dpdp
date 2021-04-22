import os
import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.utils import shuffle
import torch
from utils.data_utils import load_dataset


class DotDict(dict):
    """Wrapper around in-built dict class to access members through the dot operation.
    """

    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


def make_instance(args):
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size
    }


class VRPReader(object):
    """Iterator that reads VRP dataset files and yields mini-batches.

    Format as used in https://github.com/wouterkool/attention-learn-to-route
    """

    def __init__(self, num_nodes, num_neighbors, batch_size, filepath, target_filepath=None, do_shuffle=False):
        """
        Args:
            num_nodes: Number of nodes in VRP tours (excl depot)
            num_neighbors: Number of neighbors to consider for each node in graph
            batch_size: Batch size
            filepath: Path to dataset file (.txt file)
            variant: 'routes' to predict all edges for routes, 'clusters' to predict which nodes go together in clusters
        """
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.filepath = filepath
        filedata = load_dataset(filepath)

        self.target_filepath = target_filepath
        if target_filepath is not None:
            self.has_target = True
            target_filedata, parallelism = load_dataset(target_filepath)
            self.filedata = list([(inst, sol) for inst, sol in zip(filedata, target_filedata) if sol is not None])
        else:
            self.has_target = False
            self.filedata = list([(inst, None) for inst in filedata])

        if do_shuffle:
            self.shuffle()

        self.max_iter = (len(self.filedata) // batch_size)
        assert self.max_iter > 0, "Not enough instances ({}) for batch size ({})".format(len(self.filedata), batch_size)

    def shuffle(self):
        self.filedata = shuffle(self.filedata)  # Always shuffle upon reading data

    def __iter__(self):
        for batch in range(self.max_iter):
            start_idx = batch * self.batch_size
            end_idx = (batch + 1) * self.batch_size
            yield self.process_batch(self.filedata[start_idx:end_idx])

    def process_batch(self, batch):
        """Helper function to convert raw lines into a mini-batch as a DotDict.
        """
        batch_edges = []
        batch_edges_values = []
        batch_edges_target = []  # Binary classification targets (0/1)
        batch_nodes = []
        # batch_nodes_target = []  # Multi-class classification targets (`num_nodes` classes)
        batch_nodes_coord = []
        batch_nodes_demand = []
        batch_tour_nodes = []
        batch_tour_len = []
        # batch_route_idx_per_node = []

        # for line_num, line in enumerate(lines):
        for instance, sol in batch:
            # instance = make_instance()
            depot, loc, demand, capacity, *rest = instance
            grid_size = 1
            if len(rest) > 0:
                depot_types, customer_types, grid_size = rest

            loc = np.array(loc, dtype=float) / grid_size
            demand = np.array(demand, dtype=float) / capacity
            depot = np.array(depot, dtype=float) / grid_size

            num_nodes = len(loc)
            assert num_nodes == self.num_nodes

            # loc = torch.tensor(loc, dtype=torch.float) / grid_size
            # demand = torch.tensor(demand, dtype=torch.float) / capacity
            # depot = torch.tensor(depot, dtype=torch.float) / grid_size
            #
            # loc_with_depot = torch.cat((depot[:, None], loc))
            loc_with_depot = np.concatenate((depot[None], loc), 0)
            #             line = line.split(" ")  # Split into list

            # Compute signal on nodes
            nodes = np.zeros(num_nodes + 1)
            nodes[0] = 1  # Special token for depot

            # Convert node coordinates to required format
            #             nodes_coord = []
            #             for idx in range(0, 2 * num_nodes, 2):
            #                 nodes_coord.append([float(line[idx]), float(line[idx + 1])])

            # Compute distance matrix
            W_val = squareform(pdist(loc_with_depot, metric='euclidean'))

            # Compute adjacency matrix
            if self.num_neighbors == -1:
                W = np.ones((num_nodes + 1, num_nodes + 1))  # Graph is fully connected
            else:
                # TODO how should we deal with the depot in knn graph?
                #
                W = np.zeros((num_nodes + 1, num_nodes + 1))
                # Determine k-nearest neighbors for each node
                knns = np.argpartition(W_val, kth=self.num_neighbors, axis=-1)[:, self.num_neighbors::-1]
                # Make connections
                for idx in range(num_nodes):
                    W[idx][knns[idx]] = 1
            np.fill_diagonal(W, 2)  # Special token for self-connections
            # Special token for depot connection (3 or 4 depending on whether it is knn
            W[:, 0] += 3
            W[0, :] += 3
            W[0, 0] = 5  # Depot self-connection

            # So we have:
            # 0: node-node
            # 1: node-node knn
            # 2: node self-loop
            # 3: node-depot
            # 4: node-depot knn
            # 5: depot self-loop


            # Convert tour nodes to required format
            # Don't add final connection for tour/cycle
            # tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]][:-1]
            # dummy for now, no supervision!

            if sol is not None:
                cost, solution, duration = sol

                # Prepend and append depot just in case (duplicate depot adds no distance)
                tour_nodes = np.array([0] + solution + [0])

                # Compute node and edge representation of tour + tour_len
                tour_len = 0
                # nodes_target = np.zeros(num_nodes + 1)
                edges_target = np.zeros((num_nodes + 1, num_nodes + 1))
                for idx in range(len(tour_nodes) - 1):
                    i = tour_nodes[idx]
                    j = tour_nodes[idx + 1]
                    # nodes_target[i] = idx  # node targets: ordering of nodes in tour
                    edges_target[i][j] = 1
                    edges_target[j][i] = 1
                    tour_len += W_val[i][j]

                # Add final connection of tour in edge target
                # nodes_target[j] = len(tour_nodes) - 1
                edges_target[j][tour_nodes[0]] = 1
                edges_target[tour_nodes[0]][j] = 1
                tour_len += W_val[j][tour_nodes[0]]

                edges_target[0][0] = 0  # In case we had padding, don't add self-edge at depot

                # batch_nodes_target.append(nodes_target)
                # batch_tour_nodes.append(tour_nodes)
                batch_edges_target.append(edges_target)
                batch_tour_len.append(tour_len)

            # Concatenate the data
            batch_edges.append(W)
            batch_edges_values.append(W_val)

            batch_nodes.append(nodes)

            batch_nodes_coord.append(loc_with_depot)
            batch_nodes_demand.append(demand)


        # From list to tensors as a DotDict
        batch = DotDict()
        batch.edges = np.stack(batch_edges, axis=0)
        batch.edges_values = np.stack(batch_edges_values, axis=0)
        batch.nodes = np.stack(batch_nodes, axis=0)
        batch.nodes_coord = np.stack(batch_nodes_coord, axis=0)
        batch.nodes_demand = np.stack(batch_nodes_demand, axis=0)
        # Pad features into nodes_coord, todo rename
        batch.nodes_coord = np.concatenate((batch.nodes_coord, np.pad(batch.nodes_demand, ((0, 0), (1, 0)))[:, :, None]), -1)

        if self.has_target:
            # batch.nodes_route_idx = np.stack(batch_route_idx_per_node, axis=0)
            batch.edges_target = np.stack(batch_edges_target, axis=0)
            # batch.nodes_target = np.stack(batch_nodes_target, axis=0)
            # batch.tour_nodes = np.stack(batch_tour_nodes, axis=0)
            batch.tour_len = np.stack(batch_tour_len, axis=0)
        return batch