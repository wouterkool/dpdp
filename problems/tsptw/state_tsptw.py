import math
import torch
from typing import NamedTuple
# from utils.boolmask import mask_long2bool, mask_long_scatter
# from utils.tensor_functions import compute_in_batches


def get_distance_matrix(loc):
    # Rounds distances to integers and performs a postprocessing step to, approximately, satisfy the triangle inequality
    # From http://myweb.uiowa.edu/bthoa/TSPTWBenchmarkDataSets.htm
    # Also in https://homepages.dcc.ufmg.br/~rfsilva/tsptw/TSPTW.zip

    # Do not use .norm() here since it is inaccurate and as we round down we may round an exact integer with inaccuracy
    # down to the integer below it, resulting in wrong results (on GPU)
    d = loc[:, :, None, :] - loc[:, None, :, :]
    dist = torch.sqrt((d * d).sum(-1).float()).long()  # round down
    n = dist.size(-1)
    for i in range(n):
        for j in range(n):
            dist[:, i, j] = torch.min(dist[:, i, j], (dist[:, i, :] + dist[:, :, j]).min(dim=-1)[0])
    return dist


class StateTSPTWInt(NamedTuple):
    # Fixed input
    timew: torch.Tensor  # Dpot + loc
    dist: torch.Tensor  # (n + 1, n + 1), rounded to integer values with triangle inequality hack
    before: torch.Tensor  # (n + 1, n + 1) indicating precedence relations (following from time windows)

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency the
    # timew, dist and before tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    t: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.timew.size(-1))  # n + 1 ! TODO look into this

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            t=self.t[key],
        )

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
        assert visited_dtype == torch.uint8, \
            "Compressed mask not yet supported for TSPTW, first check code if depot is handled correctly"
        depot = input['depot']
        loc = input['loc']
        timew = input['timew']
        assert timew.dtype == torch.int64, "Timewindows should be int64"

        batch_size, n_loc, _ = loc.size()
        dist = get_distance_matrix(torch.cat((depot[:, None, :], loc), -2))
        # before[i] gives a 1 for all j that must come before i, so before[i, j] == 1 means that j comes before i!!
        before = ((timew[:, :, None, 0] + dist) > timew[:, None, :, 1])
        return StateTSPTWInt(
            timew=timew,
            dist=dist,
            before=before,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently (if there is an action for depot)
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device, dtype=torch.int64),
            t=torch.zeros(batch_size, 1, device=loc.device, dtype=torch.int64),  # Times are integers here
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()
        # assert self.visited_.
        # We are at the depot so no need to add remaining distance
        return torch.where(
            self.prev_a == 0,  # If prev_a == 0, we have visited the depot prematurely and the solution is infeasible
            self.lengths.new_tensor(math.inf, dtype=torch.float),
            (self.lengths + self.dist[self.ids, self.prev_a, 0]).float()  # Return to first step which is always 0
        )

    def update(self, selected):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected

        # Add the length (only needed for DP)
        d = self.dist[self.ids, self.prev_a, prev_a]

        # Compute new time
        lb, ub = torch.unbind(self.timew[self.ids, prev_a], -1)
        t = torch.max(self.t + d, lb)
        assert (t <= ub).all()

        # Compute lengths (costs is equal to length)
        lengths = self.lengths + d  # (batch_dim, 1)

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        return self._replace(
            prev_a=prev_a, visited_=visited_,
            lengths=lengths, t=t, i=self.i + 1
        )

    def all_finished(self):
        # Exactly n steps since node 0 depot is visited before the first step already
        return self.i.item() >= self.timew.size(-2) - 1

    def get_current_node(self):
        """
        Returns the current node where 0 is depot, 1...n are nodes
        :return: (batch_size, num_steps) tensor with current nodes
        """
        return self.prev_a

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """

        implementation = 'test2_separate'

        MAX_MASK_BATCH_SIZE = 2 if implementation != 'feasible_only' else 2 ** 32 - 1

        return compute_in_batches(lambda b: b._get_mask(implementation), MAX_MASK_BATCH_SIZE, self, n=len(self.ids))

    def _get_mask(self, implementation):

        # A node can NOT be visited if it is already visited or arrival there directly would not be within time interval
        # Additionally 1: check BEFORE conditions (all befores must be visited)
        # Additionally 2: check 1 step ahead

        # Note: this always allows going to the depot, but that should always be suboptimal so be ok
        # Cannot visit if already visited or if length that would be upon arrival is too large to return to depot
        # If the depot has already been visited then we cannot visit anymore (and this solution is infeasible!)
        visited_ = self.visited


        # x = (
        #          (
        #                  (self.t[:, :, None, None] + self.dist[self.ids, self.prev_a, :,
        #                                              None] + self.dist[self.ids, :, :])
        #                  > self.timew[self.ids, None, :, 1]
        #          ) | self.before[self.ids]
        #  ) & (visited_ == 0)[:, :, None, :]
        # print("size", x.size(), x.view(-1).size(0), x.view(-1).size(0) / (1024 * 1024))
        verbose = False
        if verbose:
            from utils.profile import mem_report_cache
            print("Mem report before")
            mem_report_cache()

        if implementation == 'feasible_only':
            mask = (
                    visited_ | visited_[:, :, 0:1] |
                    # Check that time upon arrival is OK (note that this is implied by test3 as well)
                    (self.t[:, :, None] + self.dist[self.ids, self.prev_a, :] > self.timew[self.ids, :, 1])
            )
        elif implementation == 'no_test3':
            mask = (
                    visited_ | visited_[:, :, 0:1] |
                    # Check that time upon arrival is OK (note that this is implied by test3 as well)
                    (self.t[:, :, None] + self.dist[self.ids, self.prev_a, :] > self.timew[self.ids, :, 1]) |
                    # Check Test2 from Dumas et al. 1995: all predecessors must be visited
                    # I.e. if there is a node that is not visited and should be visited before, then it becomes infeasible
                    (((visited_ == 0)[:, :, None, :] & (self.before[self.ids])).sum(-1) > 0)
            )
        elif implementation == 'cache_before':

            mask = (
                    visited_ | visited_[:, :, 0:1] |
                    # Check Test3 from Dumas et al. 1995: via j, for all unvisited k i -> j -> k must be feasible
                    # So all k must either be visited or have t + t_ij + t_jk <= ub_k
                    # The lower bound for j is not used, if using this would make i -> j -> k infeasible then k must be before j
                    # and this is already checked by Test2
                    ((
                             (
                                     (
                                             (self.t[:, :, None, None] + self.dist[self.ids, self.prev_a, :,
                                                                         None] + self.dist[self.ids, :, :])
                                             > self.timew[self.ids, None, :, 1]
                                     ) | self.before[self.ids]
                             ) & (visited_ == 0)[:, :, None, :]
                     ).sum(-1) > 0)
            )

        elif implementation == 'compute_before':
            mask = (
                visited_ | visited_[:, :, 0:1] |
                # Check Test3 from Dumas et al. 1995: via j, there must not be some k such that i -> j -> k is infeasible
                # So all k must either be visited or have max(t + t_ij, lb_j) + t_jk < ub_k
                ((
                    (
                        (
                            torch.max(
                                self.t[:, :, None, None] + self.dist[self.ids, self.prev_a, :, None],
                                self.timew[self.ids, :, None, 0]
                            ) + self.dist[self.ids, :, :]
                        ) > self.timew[self.ids, None, :, 1]
                    ) & (visited_ == 0)[:, :, None, :]
                 ).sum(-1) > 0)
            )
        elif implementation == 'test2_separate':
            mask = (
                visited_ | visited_[:, :, 0:1] |
                # Check that time upon arrival is OK (note that this is implied by test3 as well)
                # (self.t[:, :, None] + self.dist[self.ids, self.prev_a, :] > self.timew[self.ids, :, 1]) |
                # Check Test2 from Dumas et al. 1995: all predecessors must be visited
                # I.e. if there is a node that is not visited and should be visited before, then it becomes infeasible
                (((visited_ == 0)[:, :, None, :] & (self.before[self.ids])).sum(-1) > 0) |
                # Check Test3 from Dumas et al. 1995: via j, there must not be some k such that i -> j -> k is infeasible
                # So all k must either be visited or have t + t_ij + t_jk < ub_k
                # The lower bound for j is not used, if using this would make i -> j -> k infeasible then k must be before j
                # and this is already checked by Test2
                ((
                    (
                        (self.t[:, :, None, None] + self.dist[self.ids, self.prev_a, :, None] + self.dist[self.ids, :, :])
                        > self.timew[self.ids, None, :, 1]
                    ) & (visited_ == 0)[:, :, None, :]
                ).sum(-1) > 0)
            )
        else:
            assert False, "Unkown implementation"

        if verbose:
            print("Mem report after")
            mem_report_cache()

        # Depot can always be visited, however this means that the solution is infeasible but we do to prevent nans
        # (so we do not hardcode knowledge that this is strictly suboptimal if other options are available)
        # mask[:, :, 1:].all(-1) == 0
        # If all non-depot actions are infeasible, we allow the depot to be visited. This is infeasible and therefore
        # ends the tour making it infeasible, but this way we do not get nan-problems etc. Simply we should make sure
        # in the cost function that this is accounted for
        infeas = mask[:, :, 1:].sum(-1) == mask[:, :, 1:].size(-1)
        # Allow depot as infeasible action if no other option, to prevent dead end when sampling
        # we must always have at least one feasible action
        mask[:, :, 0] = (infeas == 0)
        return mask, infeas

    def construct_solutions(self, actions):
        return actions
