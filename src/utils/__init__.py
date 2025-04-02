"""Common utility functions"""

import torch
import numpy as np
import random
import os

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_device() -> str:
    """
    Get device to use for training
    
    Returns:
        'cuda' if GPU is available, 'cpu' otherwise
    """
    return "cuda" if torch.cuda.is_available() else "cpu" 