"""
Logit Normalization Loss implementation.
This loss function normalizes logits before applying cross-entropy loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitNormLoss(nn.Module):
    """
    Logit Normalization Loss.
    Normalizes logits by their L2 norm before applying cross-entropy loss.
    """
    def __init__(self, t=1.0):
        """
        Initialize LogitNormLoss.
        
        Args:
            t: Temperature parameter for scaling normalized logits (default: 1.0)
        """
        super(LogitNormLoss, self).__init__()
        self.t = t

    def forward(self, x, target):
        """
        Forward pass of LogitNormLoss.
        
        Args:
            x: Input logits tensor
            target: Target labels
            
        Returns:
            Cross-entropy loss on normalized logits
        """
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, target)
