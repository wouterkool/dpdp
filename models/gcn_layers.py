import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter


class BatchNormNode(nn.Module):
    """Batch normalization for node features.
    """

    def __init__(self, hidden_dim):
        super(BatchNormNode, self).__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)

    def forward(self, x):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)

        Returns:
            x_bn: Node features after batch normalization (batch_size, num_nodes, hidden_dim)
        """
        # The batch norm normalizes the hidden dim over batch and node dimensions
        if x.dim() == 2:
            # If we have sparse version we have only one batch dimension
            # simply perform batch norm over this (so this normalizes over batch and node dimension)
            return self.batch_norm(x)
        x_trans = x.transpose(1, 2).contiguous()  # Reshape input: (batch_size, hidden_dim, num_nodes)
        x_trans_bn = self.batch_norm(x_trans)
        x_bn = x_trans_bn.transpose(1, 2).contiguous()  # Reshape to original shape
        # x_bn2 = self.batch_norm(x.view(-1, x.size(-1))).view_as(x)
        # assert torch.allclose(x_bn, x_bn2, atol=1e-5)
        return x_bn


class BatchNormEdge(nn.Module):
    """Batch normalization for edge features.
    """

    def __init__(self, hidden_dim):
        super(BatchNormEdge, self).__init__()
        self.batch_norm = nn.BatchNorm2d(hidden_dim, track_running_stats=False)

    def forward(self, e):
        """
        Args:
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            e_bn: Edge features after batch normalization (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        # The batch norm normalizes the hidden dim over batch and edge dimensions
        if e.dim() == 2:
            # If we have sparse version we have only one batch dimension
            # simply perform batch norm over this (so this normalizes over batch and node dimension)
            # We can use the BatchNorm2d module by inserting dummy dimensions
            return self.batch_norm(e[:, :, None, None]).view_as(e)
        e_trans = e.transpose(1, 3).contiguous()  # Reshape input: (batch_size, hidden_dim, num_nodes, num_nodes)
        e_trans_bn = self.batch_norm(e_trans)
        e_bn = e_trans_bn.transpose(1, 3).contiguous()  # Reshape to original
        return e_bn


class NodeFeatures(nn.Module):
    """Convnet features for nodes.
    
    Using `sum` aggregation:
        x_i = U*x_i +  sum_j [ gate_ij * (V*x_j) ]
    
    Using `mean` aggregation:
        x_i = U*x_i + ( sum_j [ gate_ij * (V*x_j) ] / sum_j [ gate_ij] )
    """
    
    def __init__(self, hidden_dim, aggregation="mean"):
        super(NodeFeatures, self).__init__() # We must always sum, since mean means 'weighted mean' so sum weighted messages
        self.aggregation = aggregation
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V = nn.Linear(hidden_dim, hidden_dim, True)

    def forward(self, x, edge_gate, edge_index=None):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            edge_gate: Edge gate values (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
        """

        Ux = self.U(x)  # B x V x H
        Vx = self.V(x)  # B x V x H

        if edge_index is not None:
            # Sparse version
            return self.propagate(edge_index, Ux=Ux, Vx=Vx, edge_gate=edge_gate)

        from torch.utils.checkpoint import checkpoint

        # The rest is a relatively cheap operation that uses a lot of memory
        # No it does not use a lot of memory
        use_checkpoint = False
        if use_checkpoint:
            x_new = checkpoint(self._inner, edge_gate, Ux, Vx)
        else:

            x_new = self._inner(edge_gate, Ux, Vx)
            # print("Dense x", edge_gate.size(), Ux.size(), Vx.size())
            # print(x_new.flatten()[-10:])
        return x_new

    def _inner(self, edge_gate, Ux, Vx):
        use_einsum = False
        use_matmul = False

        if use_einsum:  # Seems to use more memory
            x_add = torch.einsum('bijd,bjd->bid', edge_gate, Vx)
        elif use_matmul:  # Seems to use same memory as einsum, not much faster
            x_add = torch.matmul(
                edge_gate.unsqueeze(1).transpose(1, 4).squeeze(-1),
                Vx.unsqueeze(1).transpose(1, 3)
            ).transpose(1, 3).squeeze(1)
        else:
            Vx = Vx.unsqueeze(1)  # extend Vx from "B x V x H" to "B x 1 x V x H"
            gateVx = edge_gate * Vx  # B x V x V x H
            x_add = torch.sum(gateVx, dim=-2)
        if self.aggregation=="mean":
            x_new = Ux + x_add / (1e-20 + torch.sum(edge_gate, dim=-2))  # B x V x H
        elif self.aggregation=="sum":
            x_new = Ux + x_add  # B x V x H
        return x_new

    def message(self, edge_gate, Vx_j):
        return edge_gate * Vx_j

    def update(self, agg, Ux, edge_gate, edge_index):
        src, tgt = edge_index
        # Aggregate here exactly as in _inner. Normalizing here is more efficient than normalizing the messages.
        if self.aggregation == "mean":
            gate_sum = scatter(edge_gate, tgt, dim=0, dim_size=Ux.size(0), reduce='sum')
            return Ux + agg / (1e-20 + gate_sum)
        assert self.aggregation == "sum"
        return Ux + agg  # Skip connection


class EdgeFeatures(nn.Module):
    """Convnet features for edges.

    e_ij = U*e_ij + V*(x_i + x_j)
    """

    def __init__(self, hidden_dim, directed=False):
        super(EdgeFeatures, self).__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V = nn.Linear(hidden_dim, hidden_dim, True)
        self.W = nn.Linear(hidden_dim, hidden_dim, True) if directed else None
        
    def forward(self, x, e, edge_index=None):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        Ue = self.U(e)
        Vx = self.V(x)
        Wx = Vx if self.W is None else self.W(x)  # If self.W is none, graph is undirected
        if edge_index is not None:
            # Sparse version
            src, dst = edge_index
            Wx = Wx[dst]  # = to
            Vx = Vx[src]  # = from
        else:
            Wx = Wx.unsqueeze(1)  # Extend Wx from "B x V x H" to "B x 1 x V x H" = to
            Vx = Vx.unsqueeze(2)  # extend Vx from "B x V x H" to "B x V x 1 x H" = from

        e_new = Ue + Vx + Wx
        return e_new


class ResidualGatedGCNLayer(nn.Module):
    """Convnet layer with gating and residual connection.
    """

    def __init__(self, hidden_dim, aggregation="sum", directed=False):
        super(ResidualGatedGCNLayer, self).__init__()
        self.node_feat = NodeFeatures(hidden_dim, aggregation)
        self.edge_feat = EdgeFeatures(hidden_dim, directed)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormEdge(hidden_dim)

    def forward(self, x, e, edge_index=None):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        e_in = e
        x_in = x
        # Edge convolution
        e_tmp = self.edge_feat(x_in, e_in, edge_index)  # B x V x V x H
        # Compute edge gates
        edge_gate = F.sigmoid(e_tmp)
        # Node convolution
        x_tmp = self.node_feat(x_in, edge_gate, edge_index)
        # Batch normalization
        e_tmp = self.bn_edge(e_tmp)
        x_tmp = self.bn_node(x_tmp)
        # ReLU Activation
        e = F.relu(e_tmp)
        x = F.relu(x_tmp)
        # Residual connection
        x_new = x_in + x
        e_new = e_in + e
        return x_new, e_new


class MLP(nn.Module):
    """Multi-layer Perceptron for output prediction.
    """

    def __init__(self, hidden_dim, output_dim, L=2):
        super(MLP, self).__init__()
        self.L = L
        U = []
        for layer in range(self.L - 1):
            U.append(nn.Linear(hidden_dim, hidden_dim, True))
        self.U = nn.ModuleList(U)
        self.V = nn.Linear(hidden_dim, output_dim, True)

    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, hidden_dim)

        Returns:
            y: Output predictions (batch_size, output_dim)
        """
        Ux = x
        for U_i in self.U:
            Ux = U_i(Ux)  # B x H
            Ux = F.relu(Ux)  # B x H
        y = self.V(Ux)  # B x O
        return y
