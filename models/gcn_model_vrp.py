import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.gcn_layers import ResidualGatedGCNLayer, MLP


class ResidualGatedGCNModelVRP(nn.Module):
    """Residual Gated GCN Model for outputting predictions as edge adjacency matrices.

    References:
        Paper: https://arxiv.org/pdf/1711.07553v2.pdf
        Code: https://github.com/xbresson/spatial_graph_convnets
    """

    def __init__(self, config, dtypeFloat, dtypeLong):
        super(ResidualGatedGCNModelVRP, self).__init__()
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
        self.num_segments_checkpoint = config.get('num_segments_checkpoint', 0)
        # Node and edge embedding layers/lookups
        self.nodes_coord_embedding = nn.Linear(self.node_dim, self.hidden_dim // 2, bias=False)
        # self.nodes_coord_embedding2 = nn.Linear(self.node_dim, self.hidden_dim, bias=False)
        self.edges_values_embedding = nn.Linear(1, self.hidden_dim // 2, bias=False)
        # self.edges_values_embedding2 = nn.Linear(1, self.hidden_dim, bias=False)
        self.edges_embedding = nn.Embedding(self.voc_edges_in, self.hidden_dim // 2)
        # self.edges_embedding2 = nn.Embedding(self.voc_edges_in, self.hidden_dim)
        self.nodes_embedding = nn.Embedding(self.voc_nodes_in, self.hidden_dim // 2)
        # self.nodes_embedding2 = nn.Embedding(self.voc_nodes_in, self.hidden_dim)
        # Define GCN Layers
        gcn_layers = []
        for layer in range(self.num_layers):
            gcn_layers.append(ResidualGatedGCNLayer(self.hidden_dim, self.aggregation))
        self.gcn_layers = nn.ModuleList(gcn_layers)
        # Define MLP classifiers
        self.mlp_edges = MLP(self.hidden_dim, self.voc_edges_out, self.mlp_layers)
        # self.mlp_nodes = MLP(self.hidden_dim, self.voc_nodes_out, self.mlp_layers)

    def forward(self, x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges=None, edge_cw=None):
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
        ## Todo: fix this but gives bugs for now
        x_vals = self.nodes_coord_embedding(x_nodes_coord)  # B x V x H
        x_tags = self.nodes_embedding(x_nodes)
        x = torch.cat((x_vals, x_tags), -1)
        # x = self.nodes_embedding2(x_nodes)
        e_vals = self.edges_values_embedding(x_edges_values.unsqueeze(3))  # B x V x V x H
        e_tags = self.edges_embedding(x_edges)  # B x V x V x H
        e = torch.cat((e_vals, e_tags), -1)
        #e = self.edges_values_embedding2(x_edges_values.unsqueeze(3))
        # GCN layers
        if self.num_segments_checkpoint != 0:
            layer_functions = [lambda args: layer(*args) for layer in self.gcn_layers]
            x, e = torch.utils.checkpoint.checkpoint_sequential(layer_functions, self.num_segments_checkpoint, (x, e))
        else:
            for layer in range(self.num_layers):
                # B x V x H, B x V x V x H
                x, e = self.gcn_layers[layer](x, e)
        # MLP classifier
        y_pred_edges = self.mlp_edges(e)  # B x V x V x voc_edges_out
        # y_pred_nodes = self.mlp_nodes(x)  # B x V x voc_nodes_out

        #loss = loss_edges(y_pred_edges, y_edges, edge_cw)
        # Edge loss
        y = F.log_softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
        # For some reason we must make things contiguous to prevent errors during backward
        y_perm = y.permute(0, 3, 1, 2).contiguous()  # B x voc_edges x V x V

        if y_edges is not None:
            # Compute loss
            edge_cw = torch.Tensor(edge_cw).type(self.dtypeFloat)  # Convert to tensors
            loss = nn.NLLLoss(edge_cw)(y_perm, y_edges)
        else:
            loss = None
        
        return y_pred_edges, loss
