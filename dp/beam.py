import numpy as np
import torch

from dp.potentials import update_potential_info
from utils.mask_utils import unpack_mask

from utils.sorting_utils import unique_inverse, unique1d
from utils.scatter_utils import coo_to_csr


class Beam:

    def __init__(self, num_nodes, start_node=0, mask_dtype=torch.long, cost_dtype=torch.float, device=None, gather_mask='lazy', sort_by='group_idx', columns_per_group=None, score_dtype=None, vehicle_capacity=None, potential_info=None, batch_size=1):
        assert start_node == 0, "Only support start node 0 for now, reorder for different start node"
        self.batch_ids = torch.arange(batch_size, device=device) # 0 if batch_size == 1 else
        self.batch_size = batch_size
        self.is_batch = batch_size > 1
        # # batch_size = 1
        # if torch.is_tensor(num_nodes):
        #     # We have a batch of graphs
        #     batch_size = len(num_nodes)
        #     num_nodes = num_nodes.max()
        #     batch_ids = torch.arange(batch_size, out=num_nodes.new())

        self.num_nodes = num_nodes
        self.mask_bits = torch.iinfo(mask_dtype).bits
        # assert mask_bits == 8, "Only support 8 bit masks for now"
        self.mask_num_cols = (num_nodes + self.mask_bits - 1) // self.mask_bits  # ceil
        self.columns_per_group = columns_per_group or self.mask_num_cols
        # Note: we want the mask to have contiguous columns for efficiency so we initialize and transpose
        # This has no effect currently since we initialize with only one row but this serves as a reminder
        # We group the columns in group so we limit the max memory at once
        self.col_idx_per_group = torch.arange(self.mask_num_cols, dtype=torch.long, device=device).split(self.columns_per_group)
        # We keep the mask as a list of column_groups where each column_group is size (beam_size, num_cols) but underlying memory is transposed for efficiency of finding unique
        self.mask = [torch.zeros(len(col_group), batch_size, dtype=mask_dtype, device=device) for col_group in self.col_idx_per_group]
        if vehicle_capacity is None:  # Not for VRP!
            self.mask[0][0].fill_(1)  # Start node at 1 which mask in the bitorder little is represented as 0000001 = 1 (first column of first colgroup)
        self.mask_idx = None  # Mask aligns with rows
        self.group_idx = self.batch_ids  # Different graphs must be different groups! #torch.zeros(batch_size, dtype=torch.long, device=device)
        self.current = torch.zeros(batch_size, dtype=torch.long, device=device)
        if sort_by == 'current' and len(self.batch_ids) == 1:
            # CSR representation does not work with batching
            self.current_csr = coo_to_csr(self.current, minlength=self.num_nodes)
            self.current_counts = (self.current_csr[1:] - self.current_csr[:-1])
        else:
            self.current_csr = None
            self.current_counts = None
        self.cost = torch.zeros(batch_size, dtype=cost_dtype, device=device)
        self.score = torch.zeros(batch_size, dtype=score_dtype, device=device) if score_dtype is not None else None
        self.vehicle_capacity = vehicle_capacity
        self.remaining_capacity = torch.as_tensor(vehicle_capacity, device=device).flatten() if vehicle_capacity is not None else None  # Dtype inferred from capacity
        self.start_node = start_node
        if vehicle_capacity is not None:
            # We don't want to rely on current node since it is undefined when we start at the depot
            self.current = self.current_csr = self.current_counts = None
        self.last_action = None
        self.parent = None
        self.gather_mask = gather_mask
        self.sort_by = sort_by
        self.potential_info = potential_info
        self.size = batch_size

    def get_mask_colgroup(self, idx, device=None, copy=False):
        mask = self.mask[idx]
        if self.mask_idx is None:
            # Lazy sort the mask
            return torch.empty_like(mask).copy_(mask) if copy else mask.to(device)
        return mask.to(device).index_select(-1, self.mask_idx)

    def update(self, mask, current, cost, parent=None, score=None, compute_unique_device=None, remaining_capacity=None, last_action=None, potential_info=None, batch_ids=0):
        # We will group all the entries in the beam by the mask (for each batch_id)

        device = mask[0].device
        mask_cols = [col for col_group in mask for col in col_group]  # Flatten all columns in the mask
        if torch.is_tensor(batch_ids) and len(batch_ids) > 0 and (batch_ids != batch_ids[0]).any():
            # Prepend batch id to mask_cols so we get unique masks per graph
            mask_cols.insert(0, batch_ids)
        group_idx, mask_idx_per_group = unique_inverse(mask_cols, return_index=True, device=compute_unique_device)
        # Get back all results to the current device
        mask = [msk.to(device) for msk in mask]
        group_idx = group_idx.to(device)
        mask_idx_per_group = mask_idx_per_group.to(device)
        mask_idx = None  # We don't need as mask_idx unless we sort by the mask 'lazily'
        current_csr = None  # We don't have a current csr representation unless we sort by current
        current_counts = None
        if self.sort_by is not None:
            if self.sort_by == 'current':
                assert self.batch_size == 1, "Sorting by current not compatible with batch ids"
                current, argsort = torch.sort(current)
                # If we sort by current, since there are only very few current we also store a compact 'csr' representation
                current_csr = coo_to_csr(current, minlength=self.num_nodes)
                current_counts = current_csr[1:] - current_csr[:-1]
            elif self.sort_by == 'group_idx':
                group_idx, argsort = torch.sort(group_idx)
            else:
                assert False, "Unknown sort by"

            parent = torch.gather(parent, 0, argsort) if parent is not None else None
            current = current if self.sort_by == 'current' else torch.gather(current, 0, argsort)
            group_idx = group_idx if self.sort_by == 'group_idx' else torch.gather(group_idx, 0, argsort)
            cost = torch.gather(cost, 0, argsort)
            if score is not None:
                score = torch.gather(score, 0, argsort)
            if remaining_capacity is not None:
                remaining_capacity = torch.gather(remaining_capacity, 0, argsort)
            if last_action is not None:
                last_action = torch.gather(last_action, 0, argsort)
            if potential_info is not None:
                potential_info = tuple(p_info.index_select(0, argsort) for p_info in potential_info)
            if self.batch_size > 1:
                assert self.sort_by == 'group_idx'
                # When sorting by group_idx should already be sorted by batch_id
                batch_ids = torch.gather(batch_ids, 0, argsort)
            # If mask_idx is None, then it means the mask is aligned with the sorting
            if self.gather_mask == 'eager':
                mask = [col_group.index_select(-1, argsort) for col_group in mask]
            elif self.gather_mask == 'lazy':
                # Lazy, will get the mask in order when needed
                # We want all entries of the same group to point to the same mask_idx so we can 'abuse' mask_idx as
                # group idx as well
                mask_idx = torch.gather(mask_idx_per_group, 0, group_idx)
            else:
                assert False, "Unknown gather mask option"

        self.mask = mask
        self.mask_idx = mask_idx
        self.group_idx = group_idx
        self.current = current
        self.current_csr = current_csr
        self.current_counts = current_counts
        self.cost = cost
        self.score = score
        self.remaining_capacity = remaining_capacity
        self.last_action = last_action if last_action is not None else current  # For TSP, last action is current
        self.potential_info = potential_info
        self.parent = parent
        self.batch_ids = batch_ids

        self.size = cost.size(0)

    def __len__(self):
        return self.size

    def summary(self):
        # if self._counts is not None:
        if self.size == 0:
            return "Beam (EMPTY size = 0)"
        _, counts_mask = unique1d(self.group_idx if self.group_idx is not None else self.mask_idx, return_counts=True)
        num_groups = counts_mask.size(0)
        max_groups = counts_mask.max()
        num_unique = (counts_mask == 1).sum().item()
        num_multiple = num_groups - num_unique
        unique_current, counts_current = unique1d(self.current, return_counts=True) if self.current is not None else (None, None)
        return (
            "Beam size {} \n".format(self.size)
            + "Singles {} ({:.1f} %), multiples {} ({:.1f} %) in {} groups of avg/max size {}/{}\n".format(
                num_unique,
                num_unique / self.size * 100,
                self.size - num_unique,
                (self.size - num_unique) / self.size * 100,
                num_multiple,
                "{:.2f}".format((self.size - num_unique) / num_multiple) if num_multiple > 0 else "-",
                max_groups if num_multiple > 0 else "-"
            )
            + ("Unique current nodes: {} with between {} - {} states".format(
                unique_current.size(0),
                counts_current.min(),
                counts_current.max()
            ) if unique_current is not None else "")

        )

    def destroy(self):
        # Release memory of all variables
        # TODO rather than destroying and making a new beam, we should simply update the beam
        self.batch_ids = None
        self.mask = None
        self.mask_idx = None
        self.size = None
        self.current = None
        self.group_idx = None
        self.cost = None
        self.score = None
        self.remaining_capacity = None
        self.last_action = None
        self.potential_info = None
        assert self.start_node == 0  # Note: we don't clear the start node as it remains unchanged (a scalar 0)
        self.parent = None


def update_beam(actions, beam, compute_unique_device, graph, parents, profiler, scores):
    if beam.score is None:
        # Score is equal to cost
        all_cost = scores
        scores = None
    if graph.is_vrp:
        # Num nodes is n, whereas we have 2n actions for direct and via depot
        all_current_node = actions % graph.num_nodes
    else:
        all_current_node = actions
    all_remaining_capacity = None
    new_potential_info = None
    all_parent_beamrows_l = parents.long()
    all_parent_batch_id = beam.batch_ids.gather(0, all_parent_beamrows_l) if graph.is_batch else 0
    if graph.is_vrp or beam.score is not None:
        # Need to compute cost again (if we did not store it as payload of the queue)

        all_current_l = all_current_node.long()

        if graph.is_vrp:
            all_remaining_capacity = torch.where(
                actions >= graph.num_nodes,  # num_nodes excludes depot, so 0 to num_nodes - 1 are direct
                graph.vehicle_capacity.gather(0, all_parent_batch_id),
                beam.remaining_capacity.gather(0, all_parent_beamrows_l)
            ) - graph.demand[all_parent_batch_id, all_current_l]
        if beam.score is not None:
            # Since we have a score, the cost is not equal to the score so we need to compute it
            # If we have potentials, we also need to recompute the score, since we want the score without potentials
            if graph.is_vrp:
                if beam.current is None:
                    # First step for vrp we don't have a current since we come from depot
                    all_cost = graph.cost_from_depot[all_parent_batch_id, all_current_l]
                    if beam.potential_info is not None:
                        scores = graph.score_from_depot[all_parent_batch_id, all_current_l]
                else:
                    prev = beam.current.gather(0, all_parent_beamrows_l).long()
                    all_cost = beam.cost.gather(0, all_parent_beamrows_l).add_(torch.where(
                        actions >= graph.num_nodes,
                        graph.cost_to_depot[all_parent_batch_id, prev].add_(
                            graph.cost_from_depot[all_parent_batch_id, all_current_l]),
                        graph.cost[all_parent_batch_id, prev, all_current_l]
                    ))
                    if beam.potential_info is not None:
                        scores = beam.score.gather(0, all_parent_beamrows_l).add_(torch.where(
                            actions >= graph.num_nodes,
                            graph.score_via_depot[all_parent_batch_id, prev, all_current_l],
                            graph.score[all_parent_batch_id, prev, all_current_l]
                        ))
            else:
                prev = beam.current.gather(0, all_parent_beamrows_l).long()
                all_cost = beam.cost.gather(0, all_parent_beamrows_l) + graph.cost[
                    all_parent_batch_id, prev, all_current_l]
                if beam.potential_info is not None:
                    scores = beam.score.gather(0, all_parent_beamrows_l) + graph.score[
                        all_parent_batch_id, prev, all_current_l]

        # Free memory
        prev = None
        all_current_l = None
    all_parent_beamrows_l = None
    # TODO make everything below here an update of the beam object
    mask = beam.mask
    mask_idx = beam.mask_idx
    col_idx_per_group = beam.col_idx_per_group
    potential_info = beam.potential_info
    beam.destroy()
    if potential_info is not None:
        # Update the potential info
        new_potential_info = update_potential_info(graph, potential_info, parents, all_current_node,
                                                   all_parent_batch_id)
        potential_info = None
    # actions = actions.long()  # Since we need it often to index, keep it as long
    new_mask = update_mask(all_current_node, parents, mask, mask_idx, col_idx_per_group)
    del mask
    del mask_idx
    profiler.log('update_mask')
    beam.update(new_mask, all_current_node, all_cost, parents, score=scores,
                compute_unique_device=compute_unique_device, remaining_capacity=all_remaining_capacity,
                last_action=actions if graph.is_vrp else None, potential_info=new_potential_info,
                batch_ids=all_parent_batch_id)

    profiler.log('update_beam')


def update_mask(all_actions, all_parent_beamrows, mask, mask_idx, col_idx_per_group):
    mask_bits = torch.iinfo(mask[0].dtype).bits

    # Now make a new beam
    # TODO out parameter
    # We need to find mask idxes for parent if mask does not align with sorting of beam
    parent_mask_idx = (all_parent_beamrows if mask_idx is None else torch.gather(mask_idx, 0, all_parent_beamrows.long())).long()  # group_idx[all_parent_beamrows]  # Is already an index
    # assert (mask[group_idx[all_parent_beamrows]] == mask[all_parent_beamrows]).all()
    # new_mask = mask[parent_mask_idx].scatter_add_(-1, a_packed_col[:, None], addbit)
    # Note: we want the underlying data to be transposed, so each column forms contiguous memory which
    # will be useful for sorting later
    # Doing it this way, we only need one temporary variable and no reallocations
    # out = mask[0].new_empty(parent_mask_idx.size(0))
    for i, col_idx in enumerate(col_idx_per_group):
        # Note cyling the memory only works if len(mask) == len(parent_mask_idx), i.e. if the mask is preallocated at maximum size
        # tmp = mask[i]
        # mask[i] = torch.index_select(mask, 0, parent_mask_idx.long(), out=out)
        # out = tmp
        # This is equivalent
        # mask[i], out = torch.index_select(mask, 0, parent_mask_idx.long(), out=out), mask[i]
        mask[i] = torch.index_select(mask[i], -1, parent_mask_idx)
    # We may even decide to move it back into the original memory but cycled 1
    # tmp = mask[0]
    # mask[0] = out.copy_(mask[0])
    # out = tmp
    # mask[0], out = out.copy_(mask[0]), mask[0]

    # new_mask = mask.new_empty(parent_mask_idx.size(0))
    # torch.index_select(mask, 0, parent_mask_idx.long(), out=new_mask)
    parent_mask_idx = None  # Release some memory

    a = all_actions.long()  # Unsqueeze so it is 2d like the mask
    a_bit = a % mask_bits
    # bitorder = 'little'
    # if bitorder == 'big':
    #     # Create scalar with first bit set, then shift this to the right the desired number
    #     addbit = (mask[0].new_tensor(1) << (mask_bits - 1)) >> a_bit
    # else:
    #     addbit = mask[0].new_tensor(1) << a_bit
    a_col = torch.floor_divide(a, mask_bits) if len(mask) > 1 or len(mask[0]) > 1 else None
    # a_packed_colgroup = None
    # if len(mask) > 1:
    #     a_packed_col = torch.floor_divide(a, mask_bits)  # Reuse variable
    #     if len(mask[0]) > 1:
    #         a_packed_colgroup = torch.floor_divide(a_packed_col, len(mask[0]))
    #         a_packed_col = a_packed_col.remainder_(len(mask[0]))
    #     else:
    #         a_packed_colgroup = a_packed_col
    #         a_packed_col = None
          # Reuse variable
    # new_mask = mask[parent_mask_idx]  # This is equivalent but much and much slower! (on cpu and not nice underlying mem)
    # Save some memory if possible since we need a temporary gather
    # gather_out = parent_mask_idx if parent_mask_idx.dtype == new_mask.dtype else None
    # We use gather + scatter since we have unique indices so we don't need atomic writes like scatter_add_
    # TODO is scatter + gather faster than scatter_add_ (since it does not need to reduce)?
    # new_mask = new_mask.scatter_(-1, a_packed_col, new_mask.gather(-1, a_packed_col).bitwise_or_(addbit))
    for i, (mask_col, col_idx) in enumerate(zip(mask, col_idx_per_group)):
        # TODO this is extremely inefficient
        # mask_col[a_packed_col == i] = mask_col[a_packed_col == i].bitwise_or(addbit[a_packed_col == i])
        # This is better
        # Zero (false) will remain zero when bitshift but 1s (true) get shift to the right position
        # msk indicates where the mask is updated
        msk = col_idx[:, None] == a_col[None, :] if a_col is not None else a_bit.new_tensor(1)
        # if a_packed_col is None:
        #     msk_change = a_bit.new_tensor(1) if a_packed_colgroup is None else (a_packed_colgroup == i)[None, :]
        # else:
        #     msk_change = torch.arange(mask_col.size(0), device=mask_col.device)[:, None] == a_packed_col[None, :]
        #     if a_packed_colgroup is not None:
        #         msk_change.bitwise_and_((a_packed_colgroup == i)[None, :])
        mask_col.bitwise_or_(msk << a_bit)
        del msk  # Release some mem
        # if len(mask_col) == 1:
        #     mask_col.bitwise_or_(((a_packed_colgroup == i) if a_packed_colgroup is not None else a_bit.new_tensor(1)) << a_bit)
        # else:

    # new_mask = new_mask.scatter_(-1, a_packed_col, torch.gather(new_mask, -1, a_packed_col).bitwise_or_(addbit))

    do_checks = False
    if do_checks:
        # Check if everything is correct compared to setting explicitly in dense mask
        densemaskcheck = unpack_mask(
            mask[mask_idx[all_parent_beamrows] if mask_idx is not None else all_parent_beamrows])
        densemaskcheck[np.arange(len(unpack_mask)), all_actions.numpy()] = 1
        densemask = unpack_mask(mask)

        assert (densemask == densemaskcheck).all()
    return mask