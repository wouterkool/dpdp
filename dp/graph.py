import math
import torch
import torch.nn.functional as F

from utils.mask_utils import packmask
from utils.scatter_utils import csr_to_counts, counts_to_csr, csr_to_coo, coo_to_csr


class MergedGraph:
    """
    Class that combines multiple graphs into a single batch object, by padding smaller graphs to the largest graph
    """

    def __init__(self, graphs):

        device = graphs[0].cost.device

        def get_props(name):
            return [getattr(graph, name) for graph in graphs]

        def get_props_as_tensor(name):
            return torch.tensor(get_props(name), device=device)

        def get_stacked_props(name):
            props = get_props(name)
            if torch.is_tensor(props[0]):
                dim = props[0].ndim
                assert dim <= 2
                if dim == 2:
                    max_rows = max(mat.size(0) for mat in props)
                    max_cols = max(mat.size(1) for mat in props)
                    props = [F.pad(mat, (max_cols - mat.size(1), 0, max_rows - mat.size(0), 0)) for mat in props]
                elif dim == 1:
                    max_size = max(vec.size(0) for vec in props)
                    props = [F.pad(vec, (max_size - vec.size(0), 0)) for vec in props]

                return torch.stack(props, 0)
            elif props[0] is None:
                assert all([p is None for p in props])
                return None
            return props

        self.is_batch = True
        self.batch_size = len(graphs)
        self.quantize_cost_ub = get_stacked_props('quantize_cost_ub')
        self.quantize_score_ub = get_stacked_props('quantize_score_ub')
        self.batch_num_nodes = get_props_as_tensor('num_nodes')
        self.num_nodes = self.batch_num_nodes.max().item()
        self.num_edges = get_props_as_tensor('num_edges')
        self.cost = get_stacked_props('cost')
        self.score = get_stacked_props('score')
        self.adj = get_stacked_props('adj')
        self.adj_in_packed = get_stacked_props('adj_in_packed')
        self.adj_out_packed = get_stacked_props('adj_out_packed')

        # num_nodes   = [2,    3,       3]
        # node_csr    = [0,    3,       5,      8]
        # node_offset = [0,    3,       5]
        # node_coo    = [0, 0, 1, 1, 1, 2, 2, 2]
        # nodes       = [0, 1, 0, 1, 2, 0, 1, 2]
        self.node_csr = counts_to_csr(self.batch_num_nodes) # F.pad(torch.cumsum(self.num_nodes, 0), (1, 0))
        self.node_offset = self.node_csr[:-1]
        self.node_coo = csr_to_coo(self.node_csr)
        self.nodes = torch.arange(self.batch_num_nodes.sum(), dtype=torch.long, device=device) - self.node_offset[self.node_coo]

        self.edge_weight = get_stacked_props('edge_weight')
        self.total_edge_weight_in = get_stacked_props('total_edge_weight_in')
        self.total_edge_weight_out = get_stacked_props('total_edge_weight_out')

        self.is_vrp = graphs[0].is_vrp
        if self.is_vrp:
            assert all(get_props('is_vrp'))
            self.demand = get_stacked_props('demand')
            self.vehicle_capacity = get_props_as_tensor('vehicle_capacity')
            self.cost_incl_depot = get_stacked_props('cost_incl_depot')
            self.cost_from_depot = get_stacked_props('cost_from_depot')
            self.cost_to_depot = get_stacked_props('cost_to_depot')
            self.score_from_depot = get_stacked_props('score_from_depot')
            self.score_via_depot = get_stacked_props('score_via_depot')
            self.adj_in_depot = get_stacked_props('adj_in_depot')
            self.adj_in_depot_packed = get_stacked_props('adj_in_depot_packed')
            self.adj_in_depot_list = get_stacked_props('adj_in_depot_list')
            self.adj_out_depot = get_stacked_props('adj_out_depot')
            self.adj_out_depot_packed = get_stacked_props('adj_out_depot_packed')
            self.adj_out_depot_list = get_stacked_props('adj_out_depot_list')
            self.from_depot_edge_weight = get_stacked_props('from_depot_edge_weight')
            self.to_depot_edge_weight = get_stacked_props('to_depot_edge_weight')

    def quantize_score(self, score):
        return score if self.quantize_score_ub is None else quantize(score, self.quantize_score_ub, self.score.dtype)

    def dequantize_score(self, score):
        return score if self.quantize_score_ub is None else dequantize(score.to(self.score.dtype), self.quantize_score_ub)

    def quantize_cost(self, cost):
        return cost if self.quantize_cost_ub is None else quantize(cost, self.quantize_cost_ub, self.cost.dtype)

    def dequantize_cost(self, cost):
        return cost if self.quantize_cost_ub is None else dequantize(cost.to(self.cost.dtype), self.quantize_cost_ub)


class BatchGraph:

    def __init__(self, cost, adj, score=None, quantize_cost=None, quantize_score=None, edge_weight=None):

        device = cost.device

        self.is_batch = True
        self.batch_size = cost.size(0)

        self.quantize_cost_ub = None
        if quantize_cost is not None:
            self.quantize_cost_ub, quantize_cost_dtype = quantize_cost
            cost = quantize(cost, self.quantize_cost_ub, dtype=quantize_cost_dtype)

        self.quantize_score_ub = None
        if score is not None and quantize_score is not None:
            self.quantize_score_ub, quantize_score_dtype = quantize_cost
            score = quantize(score, self.quantize_score_ub, dtype=quantize_score_dtype)

        self.cost = cost
        self.score = score
        self.adj = adj
        # self.adj_in_list = [adj_col.nonzero(as_tuple=False).flatten() for adj_col in adj.t()]
        # self.adj_out_list = [adj_row.nonzero(as_tuple=False).flatten() for adj_row in adj]
        adj_t = adj.transpose(-1, -2)
        self.adj_in_packed = packmask(adj_t.reshape(-1, adj_t.size(-1))).view(*adj_t.size()[:-1], -1)
        self.adj_out_packed = packmask(adj.reshape(-1, adj.size(-1))).view(*adj.size()[:-1], -1)
        # self.adj_sparse = self.adj.to_sparse()
        # edges_out = cost.sparse_mask(self.adj_sparse)
        # edges_in = edges_out.t().coalesce()
        # self.edge_out_cost = edges_out.values()
        # self.edge_in_cost = edges_in.values()
        # assert (cost[tuple(edges_out.indices())] == self.edge_out_cost).all()
        # assert (cost[tuple(edges_in.indices())] == self.edge_in_cost).all()
        # self.edge_out_score = self.edge_out_cost if score is None else score[tuple(edges_out.indices())]
        # self.edge_in_score = self.edge_in_cost if score is None else score[tuple(edges_in.indices())]
        # self.edge_out_idx_csr_from = coo_to_csr(edges_out.indices()[0])
        # self.edge_out_idx_to = edges_out.indices()[1]
        # self.edge_in_idx_from = edges_in.indices()[1]
        # self.edge_in_idx_csr_to = coo_to_csr(edges_in.indices()[0])
        # self.node_degree_out = csr_to_counts(self.edge_out_idx_csr_from)
        # self.node_degree_in = csr_to_counts(self.edge_in_idx_csr_to)
        self.num_nodes = cost.size(-1) # len(self.node_degree_in)
        # self.num_edges = len(self.edge_out_cost)
        # self.nodes = torch.arange(self.num_nodes, dtype=torch.long, device=device)

        # num_nodes   = [3,       3,       3]
        # node_csr    = [0,       3,       5,      8]
        # node_offset = [0,       3,       5]
        # node_coo    = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        # nodes       = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        self.batch_num_nodes = torch.full((self.batch_size, ), self.num_nodes, dtype=torch.long, device=device)
        self.node_csr = counts_to_csr(self.batch_num_nodes)
        # self.node_csr = torch.arange(self.batch_size + 1, dtype=torch.long, device=device) * self.num_nodes
        self.node_offset = self.node_csr[:-1]
        self.node_coo = csr_to_coo(self.node_csr)
        self.nodes = torch.arange(len(self.node_coo), dtype=torch.long, device=device) - self.node_offset[self.node_coo]

        self.demand = None
        self.vehicle_capacity = None
        self.cost_incl_depot = None
        self.cost_from_depot = None
        self.cost_to_depot = None
        self.score_from_depot = None
        self.score_via_depot = None
        self.adj_in_depot = None
        self.adj_in_depot_packed = None
        # self.adj_in_depot_list = None
        self.adj_out_depot = None
        self.adj_out_depot_packed = None
        # self.adj_out_depot_list = None
        self.is_vrp = False
        self.edge_weight = edge_weight
        self.total_edge_weight_in = None if edge_weight is None else edge_weight.sum(-2)
        self.total_edge_weight_out = None if edge_weight is None else edge_weight.sum(-1)
        self.from_depot_edge_weight = None
        self.to_depot_edge_weight = None

    @staticmethod
    def get_graph(
            coord, score_function='cost', heatmap=None, heatmap_threshold=1e-5, knn=None, quantize_cost_ub=None, quantize_cost_dtype=None,
            normalize='incoming', demand=None, vehicle_capacity=None, depot_penalty=math.log(0.1),
            start_node=0, node_score_weight=1.0, node_score_dist_to_start_weight=0.1
    ):
        # score_function can be cost, heatmap, heatmap_potential, heatmap_logp
        score_function_parts = score_function.split("_")
        use_logp = False
        if score_function_parts[-1] == "logp":
            use_logp = True
            score_function_parts = score_function_parts[:-1]
        add_potentials = False
        if score_function_parts[-1] == "potential":
            assert not use_logp, "Potential is not compatible with logarithmic scores"
            add_potentials = True
            score_function_parts = score_function_parts[:-1]
        assert len(score_function_parts) == 1
        score_function = score_function_parts[0]
        assert score_function in ('cost', 'heatmap')

        dist = accurate_cdist(coord, coord)

        if heatmap_threshold is not None:
            assert heatmap is not None
            log_p = heatmap
            assert (log_p <= 0).all()
            assert knn is None, "Heatmap not compatible with knn"
            adj_mask = log_p.exp() > heatmap_threshold
        elif knn is not None:
            # k-nearest neighbour
            dist_topk, _ = dist.topk(knn + 1, -1, largest=False)
            adj_mask = ((dist > 0) & (dist <= dist_topk[..., -1][..., None]))
            assert (adj_mask.sum(-1) == knn).all()
            adj_mask = adj_mask | adj_mask.transpose(-1, -2)  # Make symmetric
            if vehicle_capacity is not None:
                # For vrp, make depot always reachable
                adj_mask[:, 0, 1:] = True
                adj_mask[:, 1:, 0] = True
        else:
            # fully connected
            adj_mask = torch.eye(coord.size(-2), device=dist.device, dtype=bool).logical_not_()[None].expand_as(dist)

        edge_weight = None
        if score_function == "heatmap":
            assert heatmap is not None
            log_p = heatmap
            assert (log_p <= 0).all()

            rng_node = torch.arange(log_p.size(-1), device=log_p.device)
            # Note: in this computation we do not multiply by 'node_score'/p.max(-2) but this is almost always 1
            log_p_no_self = torch.where(
                (rng_node[None, None, :] == rng_node[None, :, None]),
                log_p.new_tensor(-math.inf),
                log_p
            )

            p = log_p_no_self.exp()
            # Maximize p so minimize -log_p or - p
            # We can't just use 1 - p to make scores positive, since then we must also make potentials 1 - potential
            # score = -log_p if use_logp else 1 - p
            score = -log_p if use_logp else -p

            if add_potentials:
                norm_dim = -1 if normalize == 'outgoing' else -2  # -1 = keep potentials based on outgoing edges, otherwise on incoming edges
                edge_perc = log_p_no_self.softmax(norm_dim)

                normalize_by_best = True
                if normalize_by_best:
                    node_score = (-p).min(norm_dim, keepdims=True).values
                else:
                    # Compute expected probability of edge as the weight/importance of the node
                    node_score = (edge_perc * -p).sum(norm_dim, keepdims=True)

                if node_score_dist_to_start_weight != 0:
                    assert demand is None or start_node == 0, "Start node must be 0 (depot) for VRP"
                    # dists_to_start = (coord[start_node, :][None, :] - coord).norm(p=2, dim=-1).unsqueeze(norm_dim)
                    dists_to_start = (coord[:, start_node, :][:, None, :] - coord).norm(p=2, dim=-1)
                    normalized_dist_to_start = dists_to_start / dists_to_start.max(-1, keepdim=True).values
                    node_score = node_score * (node_score_weight - node_score_dist_to_start_weight * (
                                normalized_dist_to_start.unsqueeze(norm_dim) - 0.5))
                else:
                    node_score = node_score * node_score_weight

                # Make edge_weight negative to align with -p since we want to maximize the potential
                edge_weight = edge_perc * node_score
        elif score_function == "cost":
            score = dist
            assert not add_potentials, "Potentials not compatible with cost"
        else:
            assert False, f"Unknown score function: {score_function}"

        quantize_cost = None
        if quantize_cost_dtype is not None:
            # Take sum of longest edges as default upper bound for (existing) route,
            # for VRP multiply by 2 as we can have more than n edges
            quantize_ub = quantize_cost_ub or (dist * adj_mask).max(-1).values.sum(-1) * (1 if demand is None else 2)

            quantize_cost = (quantize_ub, quantize_cost_dtype)
        # Lower score is better so if we want to maximize log_p we should set negative
        # Note: the edge weight is for later when we want to compute potentials, for now only use log_p
        if demand is None:
            return BatchGraph(dist, adj_mask, score=score, quantize_cost=quantize_cost, edge_weight=edge_weight)

        # VRP
        graph = BatchGraph(dist[:, 1:, 1:], adj_mask[:, 1:, 1:], score=score[:, 1:, 1:], quantize_cost=quantize_cost,
                           edge_weight=edge_weight[:, 1:, 1:] if edge_weight is not None else None)
        assert demand.numel() == len(graph.nodes)
        assert (demand > 0).all()
        assert (demand <= vehicle_capacity[:, None]).all()
        graph.demand = demand
        graph.vehicle_capacity = vehicle_capacity
        graph.cost_incl_depot = graph.quantize_cost(dist)
        graph.cost_from_depot = graph.cost_incl_depot[:, 0, 1:]
        graph.cost_to_depot = graph.cost_incl_depot[:, 1:, 0]

        if score_function == "heatmap":
            # A depot penalty of log(0.1) means we effectively multiply (in log space) the score with 0.1 to discourage going via depot
            logp_from_depot = log_p[:, 0, 1:]
            logp_to_depot = log_p[:, 1:, 0] + depot_penalty
            logp_via_depot = logp_to_depot[:, :, None] + logp_from_depot[:, None, :]
            graph.score_via_depot = graph.quantize_score(-(logp_via_depot if use_logp else logp_via_depot.exp()))
            graph.score_from_depot = graph.quantize_score(-(logp_from_depot if use_logp else logp_from_depot.exp()))
            # graph.score_to_depot = graph.quantize_score(score[1:, 0])
        elif score_function == "cost":
            # Don't use a penalty, when using the cost as score we want to be true to the cost
            graph.score_via_depot = (score[:, 1:, 0:1] + score[:, 0:1, 1:])
            graph.score_from_depot = score[:, 0, 1:]
        else:
            assert False

        graph.adj_in_depot = adj_mask[:, 1:, 0]
        graph.adj_in_depot_packed = packmask(graph.adj_in_depot)
        # graph.adj_in_depot_list = graph.adj_in_depot.nonzero(as_tuple=False).flatten()
        graph.adj_out_depot = adj_mask[:, 0, 1:]
        graph.adj_out_depot_packed = packmask(graph.adj_out_depot)
        # graph.adj_out_depot_list = graph.adj_out_depot.nonzero(as_tuple=False).flatten()
        graph.is_vrp = True
        if edge_weight is not None:
            graph.from_depot_edge_weight = edge_weight[:, 0, 1:]
            graph.to_depot_edge_weight = edge_weight[:, 1:, 0]
        return graph

    def quantize_score(self, score):
        return score if self.quantize_score_ub is None else quantize(score, self.quantize_score_ub, self.score.dtype)

    def dequantize_score(self, score):
        if self.quantize_score_ub is None:
            return score
        if not torch.is_tensor(score):
            return [dequantize(s.to(self.score.dtype), ub).item() if s is not None else None for s, ub in
                    zip(score, self.quantize_score_ub)]
        return dequantize(score.to(self.score.dtype), self.quantize_score_ub)

    def quantize_cost(self, cost):
        return cost if self.quantize_cost_ub is None else quantize(cost, self.quantize_cost_ub, self.cost.dtype)

    def dequantize_cost(self, cost):
        if self.quantize_cost_ub is None:
            return cost
        if not torch.is_tensor(cost):
            return [dequantize(c.to(self.cost.dtype), ub).item() if c is not None else None for c, ub in
                    zip(cost, self.quantize_cost_ub)]
        return dequantize(cost.to(self.cost.dtype), self.quantize_cost_ub)


class Graph:

    def __init__(self, cost, adj, score=None, quantize_cost=None, quantize_score=None, edge_weight=None):

        self.quantize_cost_ub = None
        if quantize_cost is not None:
            self.quantize_cost_ub, quantize_cost_dtype = quantize_cost
            cost = quantize(cost, self.quantize_cost_ub, dtype=quantize_cost_dtype)

        self.quantize_score_ub = None
        if score is not None and quantize_score is not None:
            self.quantize_score_ub, quantize_score_dtype = quantize_cost
            score = quantize(score, self.quantize_score_ub, dtype=quantize_score_dtype)

        self.cost = cost
        self.score = score
        self.adj = adj
        self.adj_in_list = [adj_col.nonzero(as_tuple=False).flatten() for adj_col in adj.t()]
        self.adj_out_list = [adj_row.nonzero(as_tuple=False).flatten() for adj_row in adj]
        self.adj_in_packed = packmask(adj.t())
        self.adj_out_packed = packmask(adj)
        self.adj_sparse = self.adj.to_sparse()
        edges_out = cost.sparse_mask(self.adj_sparse)
        edges_in = edges_out.t().coalesce()
        self.edge_out_cost = edges_out.values()
        self.edge_in_cost = edges_in.values()
        assert (cost[tuple(edges_out.indices())] == self.edge_out_cost).all()
        assert (cost[tuple(edges_in.indices())] == self.edge_in_cost).all()
        self.edge_out_score = self.edge_out_cost if score is None else score[tuple(edges_out.indices())]
        self.edge_in_score = self.edge_in_cost if score is None else score[tuple(edges_in.indices())]
        self.edge_out_idx_csr_from = coo_to_csr(edges_out.indices()[0])
        self.edge_out_idx_to = edges_out.indices()[1]
        self.edge_in_idx_from = edges_in.indices()[1]
        self.edge_in_idx_csr_to = coo_to_csr(edges_in.indices()[0])
        self.node_degree_out = csr_to_counts(self.edge_out_idx_csr_from)
        self.node_degree_in = csr_to_counts(self.edge_in_idx_csr_to)
        self.num_nodes = len(self.node_degree_in)
        self.num_edges = len(self.edge_out_cost)
        self.nodes = torch.arange(self.num_nodes, dtype=torch.long, device=cost.device)
        self.demand = None
        self.vehicle_capacity = None
        self.cost_incl_depot = None
        self.cost_from_depot = None
        self.cost_to_depot = None
        self.score_from_depot = None
        self.score_via_depot = None
        self.adj_in_depot = None
        self.adj_in_depot_packed = None
        self.adj_in_depot_list = None
        self.adj_out_depot = None
        self.adj_out_depot_packed = None
        self.adj_out_depot_list = None
        self.is_vrp = False
        self.edge_weight = edge_weight
        self.total_edge_weight_in = None if edge_weight is None else edge_weight.sum(-2)
        self.total_edge_weight_out = None if edge_weight is None else edge_weight.sum(-1)
        self.from_depot_edge_weight = None
        self.to_depot_edge_weight = None

    @staticmethod
    def get_graph(
            coord, score_function='cost', heatmap=None, heatmap_threshold=1e-5, knn=None, quantize_cost_ub=None, quantize_cost_dtype=None,
            normalize='incoming', demand=None, vehicle_capacity=None, depot_penalty=math.log(0.1),
            start_node=0, node_score_weight=1.0, node_score_dist_to_start_weight=0.1
    ):
        # score_function can be cost, heatmap, heatmap_potential, heatmap_logp
        score_function_parts = score_function.split("_")
        use_logp = False
        if score_function_parts[-1] == "logp":
            use_logp = True
            score_function_parts = score_function_parts[:-1]
        add_potentials = False
        if score_function_parts[-1] == "potential":
            assert not use_logp, "Potential is not compatible with logarithmic scores"
            add_potentials = True
            score_function_parts = score_function_parts[:-1]
        assert len(score_function_parts) == 1
        score_function = score_function_parts[0]
        assert score_function in ('cost', 'heatmap')

        dist = accurate_cdist(coord, coord)

        if heatmap_threshold is not None:
            assert heatmap is not None
            log_p = heatmap
            assert (log_p <= 0).all()
            assert knn is None, "Heatmap not compatible with knn"
            adj_mask = log_p.exp() > heatmap_threshold
        elif knn is not None:
            # k-nearest neighbour
            dist_topk, _ = dist.topk(knn + 1, -1, largest=False)
            adj_mask = ((dist > 0) & (dist <= dist_topk[:, -1][:, None]))
            assert (adj_mask.sum(-1) == knn).all()
            adj_mask = adj_mask | adj_mask.t()  # Make symmetric
            if vehicle_capacity is not None:
                # For vrp, make depot always reachable
                adj_mask[0, 1:] = True
                adj_mask[1:, 0] = True
        else:
            # fully connected
            adj_mask = torch.eye(coord.size(0), device=dist.device, dtype=bool).logical_not_()

        edge_weight = None
        if score_function == "heatmap":
            assert heatmap is not None
            log_p = heatmap
            assert (log_p <= 0).all()

            rng_node = torch.arange(log_p.size(-1), device=log_p.device)
            # Note: in this computation we do not multiply by 'node_score'/p.max(-2) but this is almost always 1
            log_p_no_self = torch.where(
                (rng_node[None, :] == rng_node[:, None]),
                log_p.new_tensor(-math.inf),
                log_p
            )

            p = log_p_no_self.exp()
            # Maximize p so minimize -log_p or - p
            # We can't just use 1 - p to make scores positive, since then we must also make potentials 1 - potential
            # score = -log_p if use_logp else 1 - p
            score = -log_p if use_logp else -p

            if add_potentials:
                norm_dim = -1 if normalize == 'outgoing' else -2  # -1 = keep potentials based on outgoing edges, otherwise on incoming edges
                edge_perc = log_p_no_self.softmax(norm_dim)

                normalize_by_best = True
                if normalize_by_best:
                    node_score = (-p).min(norm_dim, keepdims=True).values
                else:
                    # Compute expected probability of edge as the weight/importance of the node
                    node_score = (edge_perc * -p).sum(norm_dim, keepdims=True)

                if node_score_dist_to_start_weight != 0:
                    assert demand is None or start_node == 0, "Start node must be 0 (depot) for VRP"
                    dists_to_start = (coord[start_node, :][None, :] - coord).norm(p=2, dim=-1).unsqueeze(norm_dim)
                    node_score = node_score * (node_score_weight - node_score_dist_to_start_weight * (
                                (dists_to_start / dists_to_start.max()) - 0.5))
                else:
                    node_score = node_score * node_score_weight

                # Make edge_weight negative to align with -p since we want to maximize the potential
                edge_weight = edge_perc * node_score
        elif score_function == "cost":
            score = dist
            assert not add_potentials, "Potentials not compatible with cost"

        else:
            assert False, f"Unknown score function: {score_function}"


        quantize_cost = None
        if quantize_cost_dtype is not None:
            # Take sum of longest edges as default upper bound for (existing) route,
            # for VRP multiply by 2 as we can have more than n edges
            quantize_ub = quantize_cost_ub or (dist * adj_mask).max(-1).values.sum() * (1 if demand is None else 2)

            quantize_cost = (quantize_ub, quantize_cost_dtype)
        # Lower score is better so if we want to maximize log_p we should set negative
        # Note: the edge weight is for later when we want to compute potentials, for now only use log_p
        if demand is None:
            return Graph(dist, adj_mask, score=score, quantize_cost=quantize_cost, edge_weight=edge_weight)

        # VRP
        graph = Graph(dist[1:, 1:], adj_mask[1:, 1:], score=score[1:, 1:], quantize_cost=quantize_cost, edge_weight=edge_weight[1:, 1:] if edge_weight is not None else None)
        assert len(demand) == len(graph.nodes)
        assert (demand > 0).all()
        assert (demand <= vehicle_capacity).all()
        graph.demand = demand
        graph.vehicle_capacity = vehicle_capacity
        graph.cost_incl_depot = graph.quantize_cost(dist)
        graph.cost_from_depot = graph.cost_incl_depot[0, 1:]
        graph.cost_to_depot = graph.cost_incl_depot[1:, 0]

        if score_function == "heatmap":
            # A depot penalty of log(0.1) means we effectively multiply (in log space) the score with 0.1 to discourage going via depot
            logp_from_depot = log_p[0, 1:]
            logp_to_depot = log_p[1:, 0] + depot_penalty
            logp_via_depot = logp_to_depot[:, None] + logp_from_depot[None, :]
            graph.score_via_depot = graph.quantize_score(-(logp_via_depot if use_logp else logp_via_depot.exp()))
            graph.score_from_depot = graph.quantize_score(-(logp_from_depot if use_logp else logp_from_depot.exp()))
            # graph.score_to_depot = graph.quantize_score(score[1:, 0])
        elif score_function == "cost":
            # Don't use a penalty, when using the cost as score we want to be true to the cost
            graph.score_via_depot = (score[1:, 0:1] + score[0:1, 1:])
            graph.score_from_depot = score[0, 1:]
        else:
            assert False

        graph.adj_in_depot = adj_mask[1:, 0]
        graph.adj_in_depot_packed = packmask(graph.adj_in_depot[None]).squeeze(0)
        graph.adj_in_depot_list = graph.adj_in_depot.nonzero(as_tuple=False).flatten()
        graph.adj_out_depot = adj_mask[0, 1:]
        graph.adj_out_depot_packed = packmask(graph.adj_out_depot[None]).squeeze(0)
        graph.adj_out_depot_list = graph.adj_out_depot.nonzero(as_tuple=False).flatten()
        graph.is_vrp = True
        if edge_weight is not None:
            graph.from_depot_edge_weight = edge_weight[0, 1:]
            graph.to_depot_edge_weight = edge_weight[1:, 0]
        return graph

    def quantize_score(self, score):
        return score if self.quantize_score_ub is None else quantize(score, self.quantize_score_ub, self.score.dtype)

    def dequantize_score(self, score):
        return score if self.quantize_score_ub is None else dequantize(score.to(self.score.dtype), self.quantize_score_ub)

    def quantize_cost(self, cost):
        return cost if self.quantize_cost_ub is None else quantize(cost, self.quantize_cost_ub, self.cost.dtype)

    def dequantize_cost(self, cost):
        return cost if self.quantize_cost_ub is None else dequantize(cost.to(self.cost.dtype), self.quantize_cost_ub)


def quantize_align_batch(val, ub):
    if ub.numel() > 1:
        assert ub.size(0) == val.size(0)
        while ub.dim() < val.dim():
            ub = ub.unsqueeze(-1)
    return val, ub


def quantize(val, ub, dtype=torch.long):
    val, ub = quantize_align_batch(val, ub)
    return ((val / ub) * torch.iinfo(dtype).max).to(dtype)


def dequantize(val, ub):
    val, ub = quantize_align_batch(val, ub)
    return (val / torch.iinfo(val.dtype).max) * ub


def accurate_cdist(x1, x2):
    # Do not use matrix multiplication since this is inaccurate
    return torch.cdist(x1, x2, compute_mode='donot_use_mm_for_euclid_dist')
