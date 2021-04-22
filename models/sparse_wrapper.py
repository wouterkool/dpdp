import math
import torch
import torch.nn as nn

try:
    from torch_geometric.nn import knn_graph
except ImportError:
    knn_graph = None
    pass  # Make optional
from utils.model_utils import loss_edges


def wrap_sparse(model, type='dense'):
    if type[:3] == 'knn':
        assert knn_graph is not None, "Assert torch_geometric is installed to use knn!"
        knn_type, k = type.split("_")
        return KnnResidualGatedGCNModel(model, k=int(k), include_self_loops=knn_type=='knns')
    elif type == 'dense':
        return SparseResidualGatedGCNModel(model)
    assert False, f"Unknown sparse type: {type}"


class SparseResidualGatedGCNModel(nn.Module):

    def __init__(self, model):
        super(SparseResidualGatedGCNModel, self).__init__()
        self.model = model
        self.logit_noedge = nn.Parameter(torch.tensor([0.]))

    def forward(self, x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges= None, edge_cw=None):
        CHECK_DENSE = False
        y_edges_dense = y_edges
        if CHECK_DENSE:
            y_preds_check, loss_check = self.model(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw)

        batch_size, num_nodes = x_nodes.size()
        # Edge_idx
        edge_batch, edge_i, edge_j = self.get_edges(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw)
        node_batch, node_i = torch.ones(batch_size, num_nodes, device=x_nodes.device).nonzero(as_tuple=True)

        edge_node_offset = edge_batch * num_nodes
        edge_index = torch.stack((edge_node_offset + edge_i, edge_node_offset + edge_j), 0)
        # Now index all
        x_edges = x_edges[edge_batch, edge_i, edge_j]
        x_edges_values = x_edges_values[edge_batch, edge_i, edge_j]
        x_nodes = x_nodes[node_batch, node_i]
        x_nodes_coord = x_nodes_coord[node_batch, node_i]
        edge_cw_sparse = None
        if y_edges is not None:
            y_edges = y_edges[edge_batch, edge_i, edge_j]

            # We need to recompute the class weights for the edges considered
            # See documentation of sklearn.utils.class_weight.compute_class_weight with class_weight="balanced"
            # We used a PyTorch implementation here
            edge_label_bincount = torch.bincount(y_edges)
            num_edge_classes = len(edge_label_bincount)
            edge_cw_sparse = len(y_edges) / (num_edge_classes * edge_label_bincount)

        y_preds, loss_sparse = self.model(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw_sparse, edge_index=edge_index)

        # For now, convert predictions back to dense. For edges which are not in the graph, we always predict zero.
        y_preds_dense = y_preds.new_zeros((batch_size, num_nodes, num_nodes, y_preds.size(-1)))
        y_preds_dense[..., -1] = self.logit_noedge
        y_preds_dense[edge_batch, edge_i, edge_j] = y_preds
        if CHECK_DENSE:
            assert torch.allclose(y_preds_dense, y_preds_check, atol=1e-5)

        loss_dense = None
        if edge_cw is not None:
            if not torch.is_tensor(edge_cw):
                edge_cw = torch.Tensor(edge_cw).type(self.model.dtypeFloat)  # Convert to tensors
            loss_dense = loss_edges(y_preds_dense, y_edges_dense, edge_cw)
        return y_preds_dense, loss_dense

    def get_edges(self, x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw):
        # By default, fully connected
        batch_size, num_nodes = x_nodes.size()
        return torch.ones(batch_size, num_nodes, num_nodes, device=x_nodes.device).nonzero(as_tuple=True)


class KnnResidualGatedGCNModel(SparseResidualGatedGCNModel):

    def __init__(self, model, k=None, include_self_loops=False):
        super(KnnResidualGatedGCNModel, self).__init__(model)
        self.k = k
        self.include_self_loops = include_self_loops

    def get_edges(self, x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw):
        # By default, fully connected
        batch_size, num_nodes, _ = x_nodes_coord.size()
        batch = torch.arange(batch_size, device=x_nodes_coord.device)[:, None].repeat(1, num_nodes).flatten()
        x_nodes_coord_flat = x_nodes_coord.flatten(0, 1)
        edge_index = knn_graph(x_nodes_coord_flat, self.k, batch, loop=self.include_self_loops)
        edge_batch = edge_index[0] // num_nodes
        assert (edge_batch == edge_index[1] // num_nodes).all()
        src, tgt = edge_index % num_nodes  # Get within original graphs
        return edge_batch, src, tgt
