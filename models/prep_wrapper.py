import torch
import torch.nn as nn
import torch.nn.functional as F


class PrepWrapResidualGatedGCNModel(nn.Module):
    """
    Wrapper around model that can take raw data such that preprocessing can be done on the GPU in parallel (only for TSP)
    """

    def __init__(self, model):
        super(PrepWrapResidualGatedGCNModel, self).__init__()
        self.model = model

    def forward(self, x_nodes_coord, x_nodes_timew=None, y_tour=None, edge_cw=None):
        undirected = True
        if x_nodes_timew is not None:
            # Assume first is depot
            x_nodes = torch.zeros_like(x_nodes_coord[:, :, 0], dtype=torch.long)
            x_nodes[:, 0] = 1  # Special token for depot
            # We may need to pad the tour with a zero (depot) to learn all correct edges
            if y_tour is not None and y_tour.size(-1) == x_nodes.size(-1) - 1:
                y_tour = F.pad(y_tour, (1, 0))
            # For time windows, we want directed edges
            undirected = False
        else:
            x_nodes = torch.ones_like(x_nodes_coord[:, :, 0], dtype=torch.long)
        assert y_tour is None or y_tour.size(-1) == x_nodes.size(-1)
        x_edges_values = torch.cdist(x_nodes_coord, x_nodes_coord, compute_mode='donot_use_mm_for_euclid_dist')
        # 1 for edge, 2 for self-loop (fully connected)
        x_edges = (torch.eye(x_nodes.size(-1), dtype=torch.long, device=x_nodes.device) + 1)[None].expand_as(x_edges_values)
        y_edges = None
        if y_tour is not None:
            y_edges = torch.zeros_like(x_edges)
            y_tour_next = y_tour.roll(-1, dims=-1)
            rng = torch.arange(y_edges.size(0), device=y_edges.device)[:, None]
            y_edges[rng, y_tour, y_tour_next] = 1
            if undirected:
                y_edges[rng, y_tour_next, y_tour] = 1
        y_preds, loss = self.model(x_edges, x_edges_values, x_nodes, x_nodes_coord, x_nodes_timew=x_nodes_timew, y_edges=y_edges, edge_cw=edge_cw)
        return y_preds, loss, x_edges_values
