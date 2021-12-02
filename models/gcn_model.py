import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.gcn_layers import ResidualGatedGCNLayer, MLP
from utils.model_utils import loss_edges


class ResidualGatedGCNModel(nn.Module):
    """Residual Gated GCN Model for outputting predictions as edge adjacency matrices.

    References:
        Paper: https://arxiv.org/pdf/1711.07553v2.pdf
        Code: https://github.com/xbresson/spatial_graph_convnets
    """

    def __init__(self, config, dtypeFloat, dtypeLong):
        super(ResidualGatedGCNModel, self).__init__()
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong
        # Define net parameters
        self.num_nodes = config.num_nodes
        self.node_dim = config.node_dim
        self.voc_nodes_in = config['voc_nodes_in']
        self.voc_nodes_out = config['num_nodes']  # config['voc_nodes_out']
        self.voc_edges_in = config['voc_edges_in']
        self.voc_edges_out = config['voc_edges_out']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.mlp_layers = config['mlp_layers']
        self.aggregation = config['aggregation']
        self.directed = config.get('directed', False)
        self.num_segments_checkpoint = config.get('num_segments_checkpoint', 0)
        # Node and edge embedding layers/lookups
        self.add_node_coords = config.get('add_node_coords', True)
        if self.add_node_coords:
            self.nodes_coord_embedding = nn.Linear(self.node_dim, self.hidden_dim, bias=False)
        else:
            self.nodes_embedding = nn.Embedding(self.voc_nodes_in, self.hidden_dim)

        self.edges_values_embedding = nn.Linear(1, self.hidden_dim//2, bias=False)
        self.edges_embedding = nn.Embedding(self.voc_edges_in, self.hidden_dim//2)
        # Define GCN Layers
        gcn_layers = []
        for layer in range(self.num_layers):
            gcn_layers.append(ResidualGatedGCNLayer(self.hidden_dim, self.aggregation, self.directed))
        self.gcn_layers = nn.ModuleList(gcn_layers)
        # Define MLP classifiers
        self.mlp_edges = MLP(self.hidden_dim, self.voc_edges_out, self.mlp_layers)
        # self.mlp_nodes = MLP(self.hidden_dim, self.voc_nodes_out, self.mlp_layers)

    def forward(self, x_edges, x_edges_values, x_nodes, x_nodes_coord, x_nodes_timew=None, y_edges=None, edge_cw=None, edge_index=None):
        """
        Args:
            x_edges: Input edge adjacency matrix (batch_size, num_nodes, num_nodes)
            x_edges_values: Input edge distance matrix (batch_size, num_nodes, num_nodes)
            x_nodes: Input nodes (batch_size, num_nodes)
            x_nodes_coord: Input node coordinates (batch_size, num_nodes, node_dim)
            y_edges: Targets for edges (batch_size, num_nodes, num_nodes)
            edge_cw: Class weights for edges loss
            # y_nodes: Targets for nodes (batch_size, num_nodes, num_nodes)
            # node_cw: Class weights for nodes loss

        Returns:
            y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
            # y_pred_nodes: Predictions for nodes (batch_size, num_nodes)
            loss: Value of loss function
        """
        # Node and edge embedding
        if x_nodes_timew is None and not self.add_node_coords:
            x = self.nodes_embedding(x_nodes)
        else:
            if x_nodes_timew is None:
                x_feat = x_nodes_coord
            elif not self.add_node_coords:
                x_feat = x_nodes_timew
            else:
                x_feat = torch.cat((x_nodes_coord, x_nodes_timew), -1)
            x = self.nodes_coord_embedding(x_feat)  # B x V x H

        e_vals = self.edges_values_embedding(x_edges_values.unsqueeze(-1))  # B x V x V x H
        e_tags = self.edges_embedding(x_edges)  # B x V x V x H
        e = torch.cat((e_vals, e_tags), dim=-1)
        # GCN layers
        if self.num_segments_checkpoint != 0:
            layer_functions = [lambda inp: layer(inp[0], inp[1], edge_index) for layer in self.gcn_layers]
            x, e = torch.utils.checkpoint.checkpoint_sequential(layer_functions, self.num_segments_checkpoint, (x, e))
        else:
            for layer in range(self.num_layers):
                # B x V x H, B x V x V x H
                x, e = self.gcn_layers[layer](x, e, edge_index)
        # MLP classifier
        y_pred_edges = self.mlp_edges(e)  # B x V x V x voc_edges_out
        # y_pred_nodes = self.mlp_nodes(x)  # B x V x voc_nodes_out
        
        # Compute loss
        if y_edges is not None:
            if not torch.is_tensor(edge_cw):
                edge_cw = torch.Tensor(edge_cw).type(self.dtypeFloat)  # Convert to tensors
            loss = loss_edges(y_pred_edges, y_edges, edge_cw)
        else:
            loss = None
        
        return y_pred_edges, loss
