import torch
from torch import nn


def loss_edges(y_pred_edges, y_edges, edge_cw):
    """
    Loss function for edge predictions.

    Args:
        y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
        y_edges: Targets for edges (batch_size, num_nodes, num_nodes)
        edge_cw: Class weights for edges loss

    Returns:
        loss_edges: Value of loss function

    """
    # Edge loss
    y = torch.log_softmax(y_pred_edges, dim=-1)  # B x V x V x voc_edges
    if y.dim() > 2:
        y = y.permute(0, 3, 1, 2)  # B x voc_edges x V x V
    loss_edges = nn.NLLLoss(edge_cw)(y, y_edges)
    return loss_edges
