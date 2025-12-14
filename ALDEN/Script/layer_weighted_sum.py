"""
Layer-weighted sum implementation for combining features from multiple layers.
This code is cited from S3PRL project (https://github.com/s3prl/s3prl).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LayerWeightedSum(nn.Module):
    """
    Layer-weighted sum module for combining features from multiple layers.
    This code is cited from S3PRL project (https://github.com/s3prl/s3prl).
    """
    def __init__(self, layer_norm=True):
        """
        Initialize LayerWeightedSum.
        
        Args:
            layer_norm: Whether to apply layer normalization (default: True)
        """
        super().__init__()
        self.layer_norm = layer_norm

    def forward(self, x_list, layer_weight):
        """
        Forward pass: compute weighted sum of layer features.
        
        Args:
            x_list: List of layer representations, each is a tuple (rep_layer, _)
            layer_weight: Weight tensor for each layer
            
        Returns:
            Weighted sum of layer features
        """
        # Use list comprehension to transpose layers
        adapted_layers = [rep_layer.transpose(0, 1) for rep_layer, _ in x_list]
        n_layers = len(adapted_layers)
        stacked_hs = torch.stack(adapted_layers, dim=0)  # n_layers, B, T, C
        if self.layer_norm:
            stacked_hs = F.layer_norm(stacked_hs, (stacked_hs.shape[-1],))

        _, *origin_size = stacked_hs.size()
        stacked_hs = stacked_hs.view(n_layers, -1)  # n_layers, B * T * C
        weighted_hs = (layer_weight.unsqueeze(-1) * stacked_hs).sum(dim=0)
        weighted_hs = weighted_hs.view(*origin_size)
        return weighted_hs
