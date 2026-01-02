"""
Utility functions for setting random seeds to ensure reproducibility.
"""
import os
import random

import numpy as np
import torch


def Set_RandomSeed(random_seed):
    """
    Set the random seed for numpy, python, and cudnn to ensure reproducibility.
    
    Args:
        random_seed: Integer random seed value
    """
    # Set PyTorch random seed
    torch.manual_seed(random_seed) 
    # Set Python built-in random module seed
    random.seed(random_seed)
    # Set NumPy random seed
    np.random.seed(random_seed)
    # Set Python hash function seed for reproducible hashing
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Set random seed for all available CUDA devices
        torch.cuda.manual_seed_all(random_seed)
        # Set CuDNN (CUDA Deep Neural Network library) random number generation to deterministic
        torch.backends.cudnn.deterministic = True
        # Disable CuDNN automatic optimization mechanism
        torch.backends.cudnn.benchmark = False
