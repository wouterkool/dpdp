import torch
from torch_scatter import segment_min_coo, scatter_min

from dp.potentials import compute_vrp_expansion_solution_potentials, compute_tsp_expansion_solution_potentials
from utils.mask_utils import view_as_uint8, unpackbits, unpack_mask
from utils.scatter_utils import segment_sort_coo, segment_cummax_coo, segment_cummin_coo
from utils.sorting_utils import unique_consecutive, unique_consecutive_inverse


def get_expansions(beam, bound_first, candidate_queue, collapse, graph, use_weak_version, profiler, verbose):
    assert bound_first, "Only supports bound_first"
    assert len(beam.mask) == 1, "Only supports single colgroup for now"

    # Precompute some steps which is needed for both via depot and direct expansions
    depot_info = get_depot_info(beam, graph) if collapse and graph.is_vrp else None
    unvisited_t = beam.get_mask_colgroup(0, device=graph.cost.device, copy=True).bitwise_not_().t()
    expansion_potentials = None
    if beam.potential_info is not None:
        func = compute_vrp_expansion_solution_potentials if graph.is_vrp else compute_tsp_expansion_solution_potentials
        expansion_potentials = func(graph, unpack_mask(unvisited_t.contiguous(), num_nodes=graph.num_nodes), beam)

    max_num_uint8s = (graph.num_nodes + 7) // 8  # ceil

    if graph.is_vrp:
        add_vrp_expansion_candidates_via_depot(beam, graph, candidate_queue, depot_info, max_num_uint8s, profiler,
                                               unvisited_t, expansion_potentials=expansion_potentials, collapse=collapse)

    # In the first iteration for VRP, we can only come from (=via) depot so don't do direct expansions
    if beam.current is not None:
        add_expansion_candidates_direct(
            beam, graph, candidate_queue, depot_info, max_num_uint8s, profiler, unvisited_t,
            bound=candidate_queue.bound, expansion_potentials=expansion_potentials, collapse=collapse, use_weak_version=use_weak_version)

    device = beam.mask[0].device
    all_parent_beamrows, all_actions = candidate_queue.get_payload(device)
    all_score = candidate_queue.get_key(device)
    reduced = candidate_queue.get_num_items_reduced() or "NO"
    total_candidates = candidate_queue.total_items_queued
    profiler.log(f'get_final_topk (total {total_candidates} candidates, reduced {reduced})')
    if verbose:
        print("Total candidates considered", total_candidates)

    return all_actions, all_parent_beamrows, all_score


def get_depot_info(beam, graph):
    """
    Finds for each group (set of visited nodes) in the beam the lowest cost to return to the depot
    This is useful since any non-dominated (lowest cost) expansion via the depot must necessarily also
    arrive at the depot at lowest cost (since remaining demand is reset at depot, only look at cost)
    :param beam:
    :param graph:
    :return:
    """
    # Get total distance to depot for each entry in group, for first action current is undefined, don't add
    beam_cost_at_depot = beam.cost if beam.current is None else beam.cost + graph.cost_to_depot[
        beam.batch_ids, beam.current.long()]
    if beam.sort_by == 'group_idx':
        group_min_cost_at_depot, group_idx_min_cost_at_depot = segment_min_coo(beam_cost_at_depot, beam.group_idx)
    else:
        group_min_cost_at_depot, group_idx_min_cost_at_depot = scatter_min(beam_cost_at_depot, beam.group_idx)
    beam_min_cost_at_depot = group_min_cost_at_depot.gather(0, beam.group_idx)
    beam_idx_min_cost_at_depot = group_idx_min_cost_at_depot.gather(0, beam.group_idx)
    return group_min_cost_at_depot, group_idx_min_cost_at_depot, beam_min_cost_at_depot, beam_idx_min_cost_at_depot


def add_vrp_expansion_candidates_via_depot(beam, graph, candidate_queue, depot_info, max_num_uint8s, profiler,
                                       unvisited_t, expansion_potentials=None, collapse=True):
    # For each group, gather the inverted mask (unvisited)
    # adj_out_packed[0] is the packed adjacency of the depot
    if collapse:
        # For now we did not yet scale the implementation for VRP so we do it in one step
        group_min_cost_at_depot, group_idx_min_cost_at_depot, beam_min_cost_at_depot, beam_idx_min_cost_at_depot = depot_info

        is_feasible_via_depot = unvisited_t.index_select(0, group_idx_min_cost_at_depot).bitwise_and_(
            graph.adj_out_depot_packed[beam.batch_ids[group_idx_min_cost_at_depot]])
        col_idx, row_idx = unpackbits(view_as_uint8(is_feasible_via_depot)[:, :max_num_uint8s]).view(
            len(is_feasible_via_depot), -1).t().nonzero(as_tuple=True)

        parent_idx_expansion = group_idx_min_cost_at_depot.gather(0, row_idx)
    else:
        # Don't do inplace! Don't ruin unvisited_t!! Also, provide contiguous out parameter to save extra copy
        is_feasible_via_depot = torch.bitwise_and(unvisited_t, graph.adj_out_depot_packed[beam.batch_ids], out=unvisited_t.new_empty(*unvisited_t.size()))
        col_idx, row_idx = unpackbits(view_as_uint8(is_feasible_via_depot)[:, :max_num_uint8s]).view(
            len(is_feasible_via_depot), -1).t().nonzero(as_tuple=True)
        parent_idx_expansion = row_idx  # Since we didn't filter unvisited_t in the first place

    is_feasible_via_depot = None  # Free some memory

    # The bits were packed with bitorder little but unpacked using bitorder big, correct this by flipping the col_idx modulo 8
    expand_node_idx = col_idx.bitwise_xor_(7)
    del col_idx  # Make sure we don't use it

    parent_batch_id = beam.batch_ids[parent_idx_expansion] if graph.batch_size > 1 else 0
    if graph.score is None:
        expansion_via_depot_cost = (
            group_min_cost_at_depot.gather(0, row_idx)
            if collapse
            else beam.cost.gather(0, parent_idx_expansion).add_(beam.cost_to_depot[parent_batch_id, beam.current.long().gather(0, parent_idx_expansion)])
        ).add_(graph.cost_from_depot[parent_batch_id, expand_node_idx])
        expansion_via_depot_score = expansion_via_depot_cost
    else:

        if beam.current is None:
            # In the first iteration, the score is 0, only compute score from depot to node
            beam_best_score_at_depot = beam.score
            expansion_via_depot_score = graph.score_from_depot[parent_batch_id, expand_node_idx]
        else:
            # Note: probably order of gathering is important for efficiency
            expansion_via_depot_score = beam.score.gather(
                0, beam_idx_min_cost_at_depot.gather(0, parent_idx_expansion) if collapse else parent_idx_expansion
            ).add_(
                graph.score_via_depot[parent_batch_id, beam.current.gather(0, parent_idx_expansion).long(), expand_node_idx]
            )
            beam_best_score_at_depot = None

        if expansion_potentials is not None:  # Add potentials to score
            expansion_via_depot_score.add_(expansion_potentials[parent_idx_expansion, expand_node_idx])
    row_idx = None  # Make sure we don't use it anymore after filtering

    if candidate_queue.bound is not None:
        expansion_bound = candidate_queue.bound if graph.batch_size == 1 else candidate_queue.bound.gather(0, beam.batch_ids.gather(0, parent_idx_expansion))
        (idx,) = (expansion_via_depot_score < expansion_bound).nonzero(as_tuple=True)

        expansion_via_depot_score = torch.gather(expansion_via_depot_score, 0, idx)
        parent_idx_expansion = torch.gather(parent_idx_expansion, 0, idx)
        expand_node_idx = expand_node_idx.gather(0, idx)

    if len(parent_idx_expansion) > 0:
        expansion_via_depot_action = expand_node_idx + graph.num_nodes  # Actions n, n+1, ..., 2n-1 are via depot
        expansion_batch_ids = beam.batch_ids.gather(0, parent_idx_expansion) if graph.is_batch else 0
        candidate_queue.enqueue(expansion_via_depot_score, (parent_idx_expansion.int(), expansion_via_depot_action), batch_ids=expansion_batch_ids, already_bounded=True)
        reduced = candidate_queue.get_num_items_reduced() or "NO"
        profiler.log(f'via_depot enqueue {len(parent_idx_expansion)} candidates, reduced {reduced}')


def add_expansion_candidates_direct(beam, graph, candidate_queue, depot_info, max_num_uint8s, profiler,
                                    unvisited_t, bound=None, expansion_potentials=None, collapse=True,
                                    use_weak_version=False):
    current = beam.current.long()
    parent, action = get_unvisited_expansion_candidates(beam, current, graph, max_num_uint8s, unvisited_t)
    profiler.log(f'get {len(parent)} feasible_expansions')
    if len(parent) == 0:
        return

    parent, action, cost, batch_id = filter_feasible_expansions(
        beam, graph, current, parent, action, depot_info, bound,
        expansion_potentials, unvisited_t, max_num_uint8s, collapse, use_weak_version)
    if len(parent) == 0:
        return None
    profiler.log(f'filter {len(parent)} feasible_expansions with bound {bound if not graph.is_batch else "(batch)"}')

    if collapse:
        collapse_fn = collapse_vrp_expansions if graph.is_vrp else (collapse_tsptw_expansions if graph.has_tw else collapse_tsp_expansions)
        parent, action, cost, batch_id = collapse_fn(batch_id, beam, cost, action, graph, parent)
        profiler.log(f'collapse {len(parent)} feasible_expansions')

    score = (
        compute_expansion_scores(beam, graph, current, parent, action, batch_id, expansion_potentials)
        if beam.score is not None else
        cost
    )
    profiler.log(f'compute score')

    candidate_queue.enqueue(score, (parent.int(), action.short()), batch_ids=batch_id, already_bounded=True)
    profiler.log(f'enqueue {len(parent)} candidates, reduced {candidate_queue.get_num_items_reduced() or "NO"}')

    return parent, action, score, batch_id


def get_unvisited_expansion_candidates(beam, current, graph, max_num_uint8s, unvisited_t):
    """
    Returns the expansion node index and the idx of the entry in the beam for initial candidates:
    expansions which are unvisited and are feasible according to the adjacency graph
    :param beam:
    :param current:
    :param graph:
    :param max_num_uint8s:
    :param unvisited_t:
    :return:
    """
    is_feasible_all = graph.adj_out_packed[beam.batch_ids, current].bitwise_and_(unvisited_t)
    is_feasible_all_uint8 = view_as_uint8(is_feasible_all)[:, :max_num_uint8s]
    is_feasible_unpacked = unpackbits(is_feasible_all_uint8.flatten()).view(*is_feasible_all_uint8.size()[:-1],
                                                                            is_feasible_all_uint8.size(-1) * 8).t()
    subcol_idx, parent = is_feasible_unpacked.nonzero(as_tuple=True)
    # Since bitorder of cupy unpackbits is big by default but we use little, we need to reverse the subcol_idx,
    # by doing bitwise xor with 7 = 111 we flip all first three bits which inverts the remainder of division by 8
    # subcol_idx.bitwise_xor_(7)
    action = subcol_idx.bitwise_xor_(7)
    return parent, action


def filter_feasible_expansions(beam, graph, current, parent, action, depot_info, bound,
                               expansion_potentials, unvisited_t, max_num_uint8s, collapse, use_weak_version=False):
    """
    Filters the initial candidates in multiple ways:
    * remove expansions that would exceed the vehicle capacity (infeasible)
    * remove expansions with scores above the bound from the candidate queue (cannot be part of topk)
    * remove expansions dominated by a solution going via the depot to the same node (dominated, only if collapsing)
    * remove expansions which makes some time windows no longer feasible
    :param beam:
    :param graph:
    :param current:
    :param parent:
    :param action:
    :param depot_info:
    :param bound:
    :param expansion_potentials:
    :param collapse:
    :return:
    """
    current_feasible = torch.gather(current, 0, parent)
    batch_id_feasible = beam.batch_ids.gather(0, parent) if graph.is_batch else 0

    cost_expansions = (
        torch.gather(beam.cost, 0, parent) +
        graph.cost[batch_id_feasible, current_feasible, action]
    )

    # We're going to filter everywhere
    msk = None
    if graph.is_vrp:
        msk = beam.remaining_capacity.gather(0, parent) >= graph.demand[batch_id_feasible, action]
        if collapse:
            group_min_cost_at_depot, group_idx_min_cost_at_depot, beam_min_cost_at_depot, beam_idx_min_cost_at_depot = depot_info
            # The cost should be smaller than the cost going via the depot, since otherwise it's always better to go via depot
            msk.logical_and_(
                cost_expansions < beam_min_cost_at_depot.gather(0, parent) + graph.cost_from_depot[
                    batch_id_feasible, action])
    if graph.has_tw:
        # Check that this expansion satisfies the timewindow
        arr = (
            torch.gather(beam.time, 0, parent) +
            graph.cost[batch_id_feasible, current_feasible, action]
        )

        lb, ub = graph.timew[batch_id_feasible, action].unbind(-1)
        msk_tw = arr <= ub

        # One step lookahead, get for each current entry the earliest ending *unvisited* time window
        # Check if from each possible extension, we can still make this time window
        # We also configure a parameter to not use this much stronger extended check
        if not use_weak_version:
            unvisited_uint8 = view_as_uint8(unvisited_t.contiguous())[:, :max_num_uint8s]
            unvisited_unpacked = unpackbits(unvisited_uint8.flatten()).view(*unvisited_uint8.size()[:-1],
                                    unvisited_uint8.size(-1), 8).flip(-1).flatten(start_dim=-2)[:, :graph.num_nodes]
            first_due, idx_first_due = torch.where(
                unvisited_unpacked,
                graph.timew[beam.batch_ids, :, 1],
                graph.timew.new_tensor(torch.iinfo(graph.timew.dtype).max)
            ).min(-1)
            msk_tw.bitwise_and_(
                torch.maximum(arr, lb) + graph.cost[batch_id_feasible, action, idx_first_due.gather(0, parent).long()]
                <= first_due.gather(0, parent)
            )
            del unvisited_unpacked, unvisited_uint8

        msk = msk.logical_and_(msk_tw) if msk is not None else msk_tw
    score_expansions = None
    if bound is not None:
        # Since we're going to index anyway we apply the bound also directly
        score_expansions = (
            cost_expansions
            if beam.score is None
            else (
                torch.gather(beam.score, 0, parent) +
                graph.score[batch_id_feasible, current_feasible, action]
            )
        )
        if expansion_potentials is not None:
            assert beam.score is not None
            score_expansions.add_(expansion_potentials[parent, action])

        expansion_bound = bound if graph.batch_size == 1 else bound.gather(0, batch_id_feasible)
        msk_bound = score_expansions < expansion_bound
        msk = msk.logical_and_(msk_bound) if msk is not None else msk_bound
        del msk_bound
    current_feasible = None
    if msk is not None:
        (idx,) = msk.nonzero(as_tuple=True)
        cost_expansions = cost_expansions[idx]
        parent = parent[idx]
        batch_id_feasible = batch_id_feasible[idx] if graph.is_batch else 0
        action = action[idx]
    return parent, action, cost_expansions, batch_id_feasible


def collapse_vrp_expansions(batch_id_feasible, beam, cost_expansions, expand_node_idx, graph, idx_feasible_expansion):
    # Note: here we can use either group_idx or mask_idx but group_idx may be smaller so bit more efficient
    # TODO this can be optimized, maybe even packed in int if beam size * num_nodes <= 2^31
    idx_scatter_expansions = torch.gather(beam.group_idx, 0, idx_feasible_expansion)
    idx_scatter_expansions = (idx_scatter_expansions << 32).bitwise_or_(expand_node_idx)
    # Note: without sorted by group_idx should also be feasible by using scatter_sort
    assert beam.sort_by == 'group_idx'

    idx_scatter_expansions, group_sizes = unique_consecutive(idx_scatter_expansions, return_vals=False,
                                                             return_inverse=True, return_counts=True)
    # TODO, an optimization may be to filter groups with just 1
    # TODO max group size 64 would be more efficient but then we need to compute group sizes
    assert (cost_expansions >= 0).all()
    sorted_cost_expansions, idx_segment_sort = segment_sort_coo(cost_expansions, idx_scatter_expansions,
                                                                max_group_size=group_sizes.max())
    assert (idx_scatter_expansions[idx_segment_sort] == idx_scatter_expansions).all()
    idx_feasible_expansion = idx_feasible_expansion.gather(0, idx_segment_sort)
    expand_node_idx = expand_node_idx.gather(0, idx_segment_sort)
    remaining_capacity = beam.remaining_capacity.gather(0, idx_feasible_expansion) - graph.demand[
        batch_id_feasible, expand_node_idx]
    # Now the expansions are sorted by increasing cost, so the next expansion
    # is worse in terms of cost so is only useful if it is (strictly) better in terms of remaining capacity
    # i.e. the remaining capacity should be larger than the remaining capacity for all predecessor expansions
    # so the remaining capacity should be larger than the maximum remaining capacity for all predecessor expansions
    # we can check this using a cummax
    msk_feas = remaining_capacity >= 0
    cummax_remaining_capacity = segment_cummax_coo(remaining_capacity, idx_scatter_expansions)
    msk_feas[1:].logical_and_((idx_scatter_expansions[1:] != idx_scatter_expansions[:-1]).logical_or_(
        remaining_capacity[1:] > cummax_remaining_capacity[:-1]))
    (idx,) = msk_feas.nonzero(as_tuple=True)
    parent = torch.gather(idx_feasible_expansion, 0, idx)
    action = torch.gather(expand_node_idx, 0, idx)
    cost = torch.gather(sorted_cost_expansions, 0, idx) if beam.score is None else None
    batch_id = torch.gather(batch_id_feasible, 0, idx) if graph.is_batch else 0
    return parent, action, cost, batch_id


def collapse_tsptw_expansions(batch_id_feasible, beam, cost_expansions, expand_node_idx, graph, idx_feasible_expansion):
    # Note: here we can use either group_idx or mask_idx but group_idx may be smaller so bit more efficient
    # TODO this can be optimized, maybe even packed in int if beam size * num_nodes <= 2^31
    idx_scatter_expansions = torch.gather(beam.group_idx, 0, idx_feasible_expansion)
    idx_scatter_expansions = (idx_scatter_expansions << 32).bitwise_or_(expand_node_idx)
    # Note: without sorted by group_idx should also be feasible by using scatter_sort
    assert beam.sort_by == 'group_idx'

    idx_scatter_expansions, group_sizes = unique_consecutive(idx_scatter_expansions, return_vals=False,
                                                             return_inverse=True, return_counts=True)
    # TODO, an optimization may be to filter groups with just 1
    # TODO max group size 64 would be more efficient but then we need to compute group sizes
    assert (cost_expansions >= 0).all()
    sorted_cost_expansions, idx_segment_sort = segment_sort_coo(cost_expansions, idx_scatter_expansions,
                                                                max_group_size=group_sizes.max())
    assert (idx_scatter_expansions[idx_segment_sort] == idx_scatter_expansions).all()
    idx_feasible_expansion = idx_feasible_expansion.gather(0, idx_segment_sort)
    expand_node_idx = expand_node_idx.gather(0, idx_segment_sort)
    arr = beam.time.gather(0, idx_feasible_expansion) + graph.cost[
        batch_id_feasible, beam.current.gather(0, idx_feasible_expansion).long(), expand_node_idx]
    lb, ub = graph.timew[batch_id_feasible, expand_node_idx, :].unbind(-1)
    time = torch.maximum(arr, lb)
    # Now the expansions are sorted by increasing cost, so the next expansion
    # is worse in terms of cost so is only useful if it is (strictly) better in terms of time windows
    # i.e. the time should be earlier than the time for all predecessor expansions
    # so the time should be (strictly) smaller than the minimum time for all predecessor expansions
    # we can check this using a cummin with

    msk_feas = time <= ub  # Should be all ones if we filtered before
    cummin_t = segment_cummin_coo(time, idx_scatter_expansions)
    msk_feas[1:].logical_and_((idx_scatter_expansions[1:] != idx_scatter_expansions[:-1]).logical_or_(
        time[1:] < cummin_t[:-1]))
    (idx,) = msk_feas.nonzero(as_tuple=True)
    parent = torch.gather(idx_feasible_expansion, 0, idx)
    action = torch.gather(expand_node_idx, 0, idx)
    cost = torch.gather(sorted_cost_expansions, 0, idx) if beam.score is None else None
    batch_id = torch.gather(batch_id_feasible, 0, idx) if graph.is_batch else 0
    return parent, action, cost, batch_id


def collapse_tsp_expansions(batch_id_feasible, beam, cost_expansions, expand_node_idx, graph, idx_feasible_expansion):
    # Note: here we can use either group_idx or mask_idx but group_idx may be smaller so bit more efficient
    # TODO this can be optimized, maybe even packed in int if beam size * num_nodes <= 2^31
    idx_scatter_expansions = torch.gather(beam.group_idx, 0, idx_feasible_expansion)
    idx_scatter_expansions = (idx_scatter_expansions << 32).bitwise_or_(expand_node_idx)

    assert beam.sort_by == 'group_idx'
    idx_scatter_expansions = unique_consecutive_inverse(idx_scatter_expansions)
    cost, idx = segment_min_coo(cost_expansions, idx_scatter_expansions)
    parent = torch.gather(idx_feasible_expansion, 0, idx)
    action = torch.gather(expand_node_idx, 0, idx)
    batch_id = torch.gather(batch_id_feasible, 0, idx) if graph.is_batch else 0
    return parent, action, cost, batch_id


def compute_expansion_scores(beam, graph, current, parent, action, batch_id, expansion_potentials):
    # Compute directly from parent score and current
    action_l = action.long()
    score_expansions = torch.gather(beam.score, 0, parent) + graph.score[
        batch_id, current.gather(0, parent).long(), action_l]
    if expansion_potentials is not None:
        score_expansions.add_(expansion_potentials[parent, action_l])
    return score_expansions
