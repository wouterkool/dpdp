import os
import time
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.utils import shuffle
from utils.data_utils import load_dataset

class DotDict(dict):
    """Wrapper around in-built dict class to access members through the dot operation.
    """

    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class TSPTWReader(object):
    """Iterator that reads TSPTW dataset files and yields mini-batches.
    """

    def __init__(self, num_nodes, num_neighbors, batch_size, filepath, target_filepath=None, do_shuffle=False, do_prep=False):
        assert not do_prep, "TSPTWReader does not prepare data, use PrepWrapper"
        """
        Args:
            num_nodes: Number of nodes in TSPTW tours
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
            self.filedata = list([(inst, sol) for inst, sol in zip(filedata, target_filedata) if sol is not None and sol[1] is not None])
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
        batch_nodes_coord = []
        batch_nodes_timew = []
        batch_tour_nodes = []
        batch_tour_len = []

        # for line_num, line in enumerate(lines):
        for instance, sol in batch:
            #             line = line.split(" ")  # Split into list
            depot, loc, timew, max_coord = instance

            nodes_coord = np.concatenate((depot[None], loc), 0) / max_coord  # Normalize
            # Normalize same as coordinates to keep unit the same, not that these values do not fall in range [0,1]!
            # Todo: should we additionally normalize somehow, e.g. by expected makespan/tour length?
            nodes_timew = timew / max_coord
            # Upper bound for depot = max(node ub + dist to depot), to make this tight
            nodes_timew[0, 1] = (cdist(nodes_coord[0][None], nodes_coord[1:]) + nodes_timew[1:, 1]).max()

            # nodes_timew = nodes_timew / nodes_timew[0, 1]

            if sol is not None:
                # Convert tour nodes to required format
                # Don't add final connection for tour/cycle
                # tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]][:-1]
                # dummy for now, no supervision!
                check_cost, tour_nodes, duration = sol #list(range(len(nodes)))

                tour_len = np.array(check_cost)

                batch_tour_nodes.append(tour_nodes)
                batch_tour_len.append(tour_len)

            batch_nodes_timew.append(nodes_timew)
            batch_nodes_coord.append(nodes_coord)

        # From list to tensors as a DotDict
        batch = DotDict()
        batch.nodes_coord = np.stack(batch_nodes_coord, axis=0)
        batch.nodes_timew = np.stack(batch_nodes_timew, axis=0)
        if self.has_target:
            batch.tour_nodes = np.stack(batch_tour_nodes, axis=0)
            batch.tour_len = np.stack(batch_tour_len, axis=0)
        return batch
