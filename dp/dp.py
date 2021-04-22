import torch
from torch_scatter import segment_min_coo, segment_max_coo

from .beam import Beam, update_beam
from utils.profile_utils import Profiler
from .expansions import get_expansions
from .potentials import get_initial_potential_info
from .vrp import vrp_solution_to_actions, vrp_actions_to_solutions, get_vrp_cost


def run_dp(graph, candidate_queue, return_solution=False, collapse=True,
           mask_dtype=torch.long, beam_device=None, compute_unique_device=None, bound_first=False,
           sort_beam_by='group_idx', trace_device='cpu', enforce_return_step_feasible='hard',
           verbose=False, profile_every=-1, add_potentials=False, trace_solution=None):
    """
    Runs a dynamic program with limited size
    :param graph: A (batch) graph object
    :param candidate_queue: The queue which keeps the limited size of the candidates to expand
    :param return_solution: Whether to return the solutions (otherwise, return only final cost)
    :param collapse: Whether to remove dominated solutions. When false, this performs 'plain beam search'.
    :param mask_dtype: Dtype to use for storing the (compressed) mask of visited nodes (default 64bit long)
    :param beam_device: On which device to keep the beam (can be separate device for efficiency for large instances)
    :param compute_unique_device: On which device to perform the costly operation to identify unique masks
    :param bound_first: Whether to apply the bound from the candidate queue before collapsing dominated states.
                        Can theoretically change results but may improve performance/speed.
    :param sort_beam_by: How to sort the beam, this may affect efficiency of the implementation.
    :param trace_device: On which device to keep the trace for backtracking solutions. Default 'cpu' to save gpu memory.
    :param enforce_return_step_feasible: Whether to enforce the return to the start node to be in the adjacency graph.
    :param verbose: Show debug output
    :param profile_every: Perform profiling every n steps
    :param add_potentials: Whether to add potentials to the scoring policy
    :param trace_solution: For debugging, an existing solution to trace forward to see where it drops off the beam
    :return:
    """

    start_node = 0  # Not yet other supported
    potential_info = get_initial_potential_info(graph, start_node) if add_potentials else None

    beam = Beam(
        graph.num_nodes, start_node=start_node, mask_dtype=mask_dtype, cost_dtype=graph.cost.dtype, device=beam_device,
        sort_by=sort_beam_by, columns_per_group=1 if candidate_queue.capacity >= int(1e8) else None,
        score_dtype=graph.score.dtype if graph.score is not None else None,
        vehicle_capacity=graph.vehicle_capacity if graph.is_vrp else None,
        potential_info=potential_info, batch_size=graph.batch_size
    )

    trace = []
    for step in range(graph.num_nodes if graph.is_vrp else graph.num_nodes - 1):  # For TSP, we have a start node so only n - 1 steps
        if verbose:
            print("Step", step, "beam size", len(beam))
            print(beam.summary())
        if beam.size == 0:
            if verbose:
                print("Stopping early since no solution was found")
            break

        beam = dp_step(
            step, beam, graph, candidate_queue=candidate_queue, collapse=collapse,
            profile=profile_every > 0 and step % profile_every == 0, verbose=verbose,
            compute_unique_device=compute_unique_device, bound_first=bound_first
        )

        if return_solution:
            trace.append((beam.parent.to(trace_device), beam.last_action.to(trace_device)))

    return finalize_solutions(beam, graph, return_solution, start_node, trace, trace_device,
                              enforce_return_step_feasible, trace_solution, verbose)


def finalize_solutions(beam, graph, return_solution, start_node, trace, trace_device, enforce_return_step_feasible,
                       trace_solution, verbose):
    """
    Finalizes solutions
    :param beam: The final beam
    :param graph: The graph
    :param return_solution: Boolean whether to return solutions
    :param start_node: The start node
    :param trace: The trace
    :param trace_device: Device on which to keep the trace
    :param enforce_return_step_feasible: Whether to enforce the last step to return to be feasible,
            i.e. return edge must be in adjacency graph (formally more correct but practically worse):
        'soft' infeasible solutions only allowed if no solution on the beam can feasibly return to start
        'hard' infeasible solutions not allowed
        None: allow return to start to be infeasible, simply take best final solution
    :param trace_solution: For debugging, an existing solution to trace forward to see where it drops off the beam
    :param verbose: Show debug output
    :return: (cost, output) as a list with None for infeasible solutions, or only cost (depending on return_solution)
    """
    # Finalize
    current = beam.current.long()
    is_return_feasible = (graph.adj_in_depot if graph.is_vrp else graph.adj[:, :, start_node])[beam.batch_ids, current]
    cost_to_return = (graph.cost_to_depot if graph.is_vrp else graph.cost[:, :, start_node])[beam.batch_ids, current]
    final_cost = beam.cost + cost_to_return
    assert (beam.batch_ids[1:] >= beam.batch_ids[:-1]).all()
    min_final_cost, best_ind = segment_min_coo(final_cost, beam.batch_ids, dim_size=graph.batch_size)
    has_solution = best_ind < beam.size
    if enforce_return_step_feasible in ('soft', 'hard'):
        any_feasible = segment_max_coo(is_return_feasible.int(), beam.batch_ids, dim_size=graph.batch_size)[0].bool()
        cost_infeas = (torch.finfo if torch.is_floating_point(final_cost) else torch.iinfo)(final_cost.dtype).max
        final_cost_feasible = torch.where(is_return_feasible, final_cost, final_cost.new_tensor(cost_infeas))
        min_final_cost_feasible, best_ind_feasible = segment_min_coo(final_cost_feasible, beam.batch_ids,
                                                                     dim_size=graph.batch_size)
        if enforce_return_step_feasible == 'hard':
            has_solution = any_feasible
            min_final_cost, best_ind = min_final_cost_feasible, best_ind_feasible
        elif enforce_return_step_feasible == 'soft':
            min_final_cost = torch.where(any_feasible, min_final_cost_feasible, min_final_cost)
            best_ind = torch.where(any_feasible, best_ind_feasible, best_ind)
    # Filter only those which finally have a solution
    best_ind = best_ind[has_solution]
    min_final_cost = min_final_cost[has_solution]
    batch_ids = beam.batch_ids[best_ind]
    min_final_cost_batch = [None] * graph.batch_size
    for batch_id, cost in zip(batch_ids, min_final_cost):
        min_final_cost_batch[batch_id] = cost
    if return_solution:

        best_ind = best_ind.to(trace_device)

        # We must add one more 'expansion' to the trace to select the one entry we want to backtrack
        # For TSP, this is the return to start node action,
        # for VRP the action is go to depot which is a special action (-1)
        trace.append((best_ind, torch.full_like(best_ind, -1 if graph.is_vrp else start_node)))
        parents, actions = zip(*trace)
        solutions = backtrack(parents, actions)

        if trace_solution is not None:
            trace_actions = vrp_solution_to_actions(trace_solution, graph.num_nodes)
            fwd_trace = trace_forward(parents, actions, trace_actions)
            print("Tracing solution: ", fwd_trace)

        if graph.is_vrp:
            solution = vrp_actions_to_solutions(solutions[:, :-1], graph.num_nodes)
            check_cost = get_vrp_cost(solution.to(graph.demand.device), graph.demand[batch_ids],
                                      graph.vehicle_capacity[batch_ids], graph.cost_incl_depot[batch_ids])
        else:
            solution = solutions.roll(1, dims=-1)  # Start with start node
            assert (solution[:, 0] == start_node).all()  # Just checking!
            check_cost = graph.cost[batch_ids[:, None], solution, solution.roll(-1, dims=-1)].sum(-1)
        if verbose:
            print(solution, min_final_cost)
        assert torch.allclose(check_cost.to(min_final_cost.dtype),
                              min_final_cost), "Check cost: {} not equal to {}".format(check_cost, min_final_cost)
        solution_batch = [None] * graph.batch_size
        for batch_id, sol in zip(batch_ids, solution):
            solution_batch[batch_id] = sol
        return min_final_cost_batch, solution_batch
    return min_final_cost_batch


def dp_step(step, beam, graph, candidate_queue, collapse=True, profile=False, verbose=False, compute_unique_device=None, bound_first=False):
    profiler = Profiler(dummy=not profile, device=beam.cost.device)

    candidate_queue.reset()
    actions, parents, scores = get_expansions(beam, bound_first, candidate_queue, collapse, graph, profiler, verbose)

    update_beam(actions, beam, compute_unique_device, graph, parents, profiler, scores)

    if profile:
        profiler.print_summary(step)

    return beam


def trace_forward(parents, actions, solution):
    cur = 0
    trace = []
    for parent, action, sol in zip(parents, actions, solution):
        (match, ) = ((parent == cur) & (action == sol)).nonzero(as_tuple=True)
        if len(match) == 0:
            return trace
        cur = match.item()
        trace.append(cur)


def backtrack(parents, actions):

    # Now backtrack to find aligned action sequences in reversed order
    cur_parent = parents[-1].long()
    reversed_aligned_sequences = [actions[-1]]
    for parent, sequence in reversed(list(zip(parents[:-1], actions[:-1]))):
        reversed_aligned_sequences.append(sequence.gather(-1, cur_parent))
        cur_parent = parent.gather(-1, cur_parent).long()

    return torch.stack(list(reversed(reversed_aligned_sequences)), -1)
