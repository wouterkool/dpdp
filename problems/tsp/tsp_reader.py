import os
import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.utils import shuffle
from utils.data_utils import load_dataset

class DotDict(dict):
    """Wrapper around in-built dict class to access members through the dot operation.
    """

    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class TSPReader(object):
    """Iterator that reads TSP dataset files and yields mini-batches.

    Format as used in https://github.com/wouterkool/attention-learn-to-route
    """

    def __init__(self, num_nodes, num_neighbors, batch_size, filepath, target_filepath=None, do_shuffle=False, do_prep=True):
        """
        Args:
            num_nodes: Number of nodes in TSP tours
            num_neighbors: Number of neighbors to consider for each node in graph
            batch_size: Batch size
            filepath: Path to dataset file (.txt file)
        """
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.filepath = filepath
        filedata = load_dataset(filepath)  # open(filepath, "r").readlines()

        self.target_filepath = target_filepath
        if target_filepath is not None:
            self.has_target = True
            target_filedata, parallelism = load_dataset(target_filepath)
            self.filedata = list([(inst, sol) for inst, sol in zip(filedata, target_filedata) if sol is not None])
        else:
            self.has_target = False
            self.filedata = list([(inst, None) for inst in filedata])

        self.do_prep = do_prep

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
        batch_nodes_target = []  # Multi-class classification targets (`num_nodes` classes)
        batch_nodes_coord = []
        batch_tour_nodes = []
        batch_tour_len = []

        # for line_num, line in enumerate(lines):
        for nodes_coord, sol in batch:
            #             line = line.split(" ")  # Split into list
            if self.do_prep:
                # Compute signal on nodes
                nodes = np.ones(self.num_nodes)  # All 1s for TSP...

                # Convert node coordinates to required format
                #             nodes_coord = []
                #             for idx in range(0, 2 * self.num_nodes, 2):
                #                 nodes_coord.append([float(line[idx]), float(line[idx + 1])])

                # Compute distance matrix
                W_val = squareform(pdist(nodes_coord, metric='euclidean'))

                # Compute adjacency matrix
                if self.num_neighbors == -1:
                    W = np.ones((self.num_nodes, self.num_nodes))  # Graph is fully connected
                else:
                    W = np.zeros((self.num_nodes, self.num_nodes))
                    # Determine k-nearest neighbors for each node
                    knns = np.argpartition(W_val, kth=self.num_neighbors, axis=-1)[:, self.num_neighbors::-1]
                    # Make connections
                    for idx in range(self.num_nodes):
                        W[idx][knns[idx]] = 1
                np.fill_diagonal(W, 2)  # Special token for self-connections

            if sol is not None:
                # Convert tour nodes to required format
                # Don't add final connection for tour/cycle
                # tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]][:-1]
                # dummy for now, no supervision!
                check_cost, tour_nodes, duration = sol #list(range(len(nodes)))

                if self.do_prep:
                    # Compute node and edge representation of tour + tour_len
                    tour_len = 0
                    nodes_target = np.zeros(self.num_nodes)
                    edges_target = np.zeros((self.num_nodes, self.num_nodes))
                    for idx in range(len(tour_nodes) - 1):
                        i = tour_nodes[idx]
                        j = tour_nodes[idx + 1]
                        nodes_target[i] = idx  # node targets: ordering of nodes in tour
                        edges_target[i][j] = 1
                        edges_target[j][i] = 1
                        tour_len += W_val[i][j]

                    # Add final connection of tour in edge target
                    nodes_target[j] = len(tour_nodes) - 1
                    edges_target[j][tour_nodes[0]] = 1
                    edges_target[tour_nodes[0]][j] = 1
                    tour_len += W_val[j][tour_nodes[0]]
                    assert np.allclose(tour_len, check_cost, atol=1e-5)

                    batch_edges_target.append(edges_target)
                    batch_nodes_target.append(nodes_target)
                else:
                    tour_len = np.array(check_cost)

                batch_tour_nodes.append(tour_nodes)
                batch_tour_len.append(tour_len)

            # Concatenate the data
            if self.do_prep:
                batch_edges.append(W)
                batch_edges_values.append(W_val)
                batch_nodes.append(nodes)
            batch_nodes_coord.append(nodes_coord)

        # From list to tensors as a DotDict
        batch = DotDict()
        if self.do_prep:
            batch.edges = np.stack(batch_edges, axis=0)
            batch.edges_values = np.stack(batch_edges_values, axis=0)
            batch.nodes = np.stack(batch_nodes, axis=0)
        batch.nodes_coord = np.stack(batch_nodes_coord, axis=0)
        if self.has_target:
            if self.do_prep:
                batch.edges_target = np.stack(batch_edges_target, axis=0)
                batch.nodes_target = np.stack(batch_nodes_target, axis=0)
            batch.tour_nodes = np.stack(batch_tour_nodes, axis=0)
            batch.tour_len = np.stack(batch_tour_len, axis=0)
        return batch
