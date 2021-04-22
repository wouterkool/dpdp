import torch
from torch.nn import functional as F
from torch_scatter import segment_sum_coo


def vrp_solution_to_actions(sol, num_nodes):
    actions = sol.clone()
    actions[1:][actions[:-1] == 0] += num_nodes
    actions[0] += num_nodes
    actions = actions[actions != 0] - 1
    return actions


def vrp_actions_to_solutions(actions, num_nodes=None):
    # Convert to variable length representations
    cur_idx = torch.zeros_like(actions[:, 0])
    a_converted = torch.zeros_like(cur_idx)
    # finished = cur_idx.bool()  # hacky all false
    n_steps = actions.size(-1)
    num_nodes = num_nodes or n_steps  # If not provided, assume full solution
    a_convs = []
    while (cur_idx < n_steps).any():
        a = actions.gather(-1, torch.clamp(cur_idx, 0, n_steps - 1)[:, None])[:, 0]
        a_node = a % num_nodes
        # Needs to see a depot if a >= n (depot action) and not already satisfied
        a_depot = (((a >= num_nodes) & (a_converted != 0)) | (cur_idx >= n_steps)).long()  # a / n
        a_converted = (a_node + 1) * (1 - a_depot)  # 0 if depot or finished, else a_node + 1
        cur_idx += (1 - a_depot)  # for depot, don't shift cur idx
        a_convs.append(a_converted)
    return torch.stack(a_convs, -1)


def get_vrp_cost(solution, demand, vehicle_capacity, cost_incl_depot):
    assert (cost_incl_depot[:, 0, 0] == 0).all()
    batch_rng = torch.arange(len(solution), device=solution.device)
    cost = (
        cost_incl_depot[batch_rng, 0, solution[:, 0]] +
        cost_incl_depot[batch_rng[:, None], solution[:, :-1], solution[:, 1:]].sum(-1) +
        cost_incl_depot[batch_rng, solution[:, -1], 0]
    )
    route_idx = (solution == 0).cumsum(-1)
    demands_per_route = segment_sum_coo(F.pad(demand, (1, 0)).gather(-1, solution), route_idx)
    assert (demands_per_route <= vehicle_capacity[:, None]).all()
    # Make sure each solution visits each node exactly once
    assert solution.count_nonzero() == demand.numel()
    sol_nonz_sorted = solution[solution != 0].view(*demand.size()).sort(-1).values
    assert (sol_nonz_sorted[:, 1:] - sol_nonz_sorted[:, :-1] == 1).all()
    return cost