import torch


def compute_vrp_expansion_solution_potentials(graph, mask_unvisited, beam, add_from_current_node=False, add_to_return_node=True, DO_CHECK=False):
    # Since the expansion always has a node visited, this function also works for empty solution
    node_remaining_edge_weight_in, node_remaining_edge_weight_out, depot_remaining_edge_weight_in, depot_remaining_edge_weight_out = beam.potential_info

    if DO_CHECK:
        beam_edge_weight = graph.edge_weight[beam.batch_ids]
        node_remaining_edge_weight_in_check = (mask_unvisited[:, :, None] * beam_edge_weight).sum(-2)
        assert torch.allclose(node_remaining_edge_weight_in, node_remaining_edge_weight_in_check, atol=1e-5)

        node_remaining_edge_weight_out_check = (mask_unvisited[:, None, :] * beam_edge_weight).sum(-1)
        assert torch.allclose(node_remaining_edge_weight_out, node_remaining_edge_weight_out_check, atol=1e-5)

    node_potentials = node_remaining_edge_weight_in
    # TODO matrix mult more efficient? Or compute also incrementally
    unvisited_potentials = (node_potentials * mask_unvisited).sum(-1)

    if add_to_return_node:
        # Since the return node (the depot) is the same for each expansion, it is more efficient to add it first
        unvisited_potentials = unvisited_potentials + depot_remaining_edge_weight_in

    # Always subtract the nodes to the new current node since that will be visited after expanding
    unvisited_potentials_expansions = unvisited_potentials[:, None] - node_remaining_edge_weight_in

    if not add_from_current_node:
        # Only subtract the out weights for the new node if we don't want weights from the current node
        unvisited_potentials_expansions = unvisited_potentials_expansions - node_remaining_edge_weight_out
        if add_to_return_node:
            # We have added earlier from the new current node to the depot node so we should subtract this again
            unvisited_potentials_expansions = unvisited_potentials_expansions - graph.to_depot_edge_weight[beam.batch_ids, :]

    return unvisited_potentials_expansions


def compute_tsp_expansion_solution_potentials(graph, mask_unvisited, beam, add_from_current_node=False, add_to_return_node=True, DO_CHECK=False):
    """
    For each entry in the beam, computes the potentials for all expansions incrementally from the potential_info
    :param graph:
    :param mask_unvisited: mask of unvisited nodes
    :param beam:
    :param add_from_current_node: whether to also add potentials of edges outgoing of the current node (default False)
    :param add_to_return_node: whether to also add potentials for returning to the start node (default True)
    :param DO_CHECK: perform internal consistency checks
    :return:
    """
    assert beam.start_node is not None, "TODO: does this implementation work without start node?"
    # Since the expansion always has at least one node visited, this function could also work for empty solution

    node_remaining_edge_weight_in, node_remaining_edge_weight_out = beam.potential_info

    if DO_CHECK:
        beam_edge_weight = graph.edge_weight[beam.batch_ids]  # Add beam dimension
        node_remaining_edge_weight_in_check = (mask_unvisited[:, :, None] * beam_edge_weight).sum(-2)
        assert torch.allclose(node_remaining_edge_weight_in, node_remaining_edge_weight_in_check, atol=1e-5)

        node_remaining_edge_weight_out_check = (mask_unvisited[:, None, :] * beam_edge_weight).sum(-1)
        assert torch.allclose(node_remaining_edge_weight_out, node_remaining_edge_weight_out_check, atol=1e-5)

    node_potentials = node_remaining_edge_weight_in
    # TODO matrix mult more efficient? Or compute also incrementally (but this should not be bottleneck)
    unvisited_potentials = (node_potentials * mask_unvisited).sum(-1)

    if add_to_return_node:
        # Since the return node is the same for each expansion, it is more efficient to add it first
        # Add weights from every unvisited to return_to_node, only if the first node has been visited
        # otherwise it was not masked so is already in the potentials
        unvisited_potentials = unvisited_potentials + node_remaining_edge_weight_in[:, beam.start_node]

    # To compute potentials for expansions, we should subtract the in and out edge_weights for the expanded node (note, self edges are 0)
    # Always subtract the nodes to the new current node since that will be visited after expanding
    if beam.start_node is not None or not add_to_return_node:  # For TSP since we should return to first node still
        unvisited_potentials_expansions = unvisited_potentials[:, None] - node_remaining_edge_weight_in
    else:
        assert beam.start_node is None and add_to_return_node
        # Here we have no start node (empty) solution, so the next expansion is the first node and
        # since and we need to include weights to the return nodes, we want all weights for all expansions
        unvisited_potentials_expansions = unvisited_potentials[:, None].expand_as(node_remaining_edge_weight_in)

    if not add_from_current_node:
        # Only subtract the out weights for the new node if we don't want weights from the current node
        unvisited_potentials_expansions = unvisited_potentials_expansions - node_remaining_edge_weight_out
        # if add_to_return_node and beam.state.i.item() > 0:  # For TSP
        if add_to_return_node and beam.start_node is not None:
            # We have added earlier from the new current node to the start node so we should subtract this again
            unvisited_potentials_expansions = unvisited_potentials_expansions - graph.edge_weight[beam.batch_ids, :, beam.start_node]

    return unvisited_potentials_expansions


def update_potential_info(graph, potential_info, parent, visited_node, parent_batch_id):
    parent = parent.long()
    visited_node = visited_node.long()
    prev_node_remaining_edge_weight_in, prev_node_remaining_edge_weight_out, *vrp_potential_info = potential_info

    # Update remaining weights in by subtracting the edge_weights from the current node to each node
    node_remaining_edge_weight_in = prev_node_remaining_edge_weight_in[parent] - graph.edge_weight[parent_batch_id, visited_node, :]
    # Update remaining weights out by subtracting the edge_weights to the current node from each node
    node_remaining_edge_weight_out = prev_node_remaining_edge_weight_out[parent] - graph.edge_weight[parent_batch_id, :, visited_node]

    if not graph.is_vrp:  # TSP
        return node_remaining_edge_weight_in, node_remaining_edge_weight_out

    # VRP
    prev_depot_remaining_edge_weight_in, prev_depot_remaining_edge_weight_out = vrp_potential_info
    depot_remaining_edge_weight_in = prev_depot_remaining_edge_weight_in[parent] - graph.to_depot_edge_weight[parent_batch_id, visited_node]
    depot_remaining_edge_weight_out = prev_depot_remaining_edge_weight_out[parent] - graph.from_depot_edge_weight[parent_batch_id, visited_node]
    return node_remaining_edge_weight_in, node_remaining_edge_weight_out, depot_remaining_edge_weight_in, depot_remaining_edge_weight_out


def get_initial_potential_info(graph, start_node):
    if graph.score is None:
        return None  # Potentials only with score
    if graph.is_vrp:
        return (
            graph.total_edge_weight_in,
            graph.total_edge_weight_out,
            graph.to_depot_edge_weight.sum(-1),
            graph.from_depot_edge_weight.sum(-1)
        )
    return (  # TSP
        # Subtract start node which is already visited in beam
        graph.total_edge_weight_in - graph.edge_weight[:, start_node, :],
        graph.total_edge_weight_out - graph.edge_weight[:, :, start_node]
    )