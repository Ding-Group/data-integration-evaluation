import random

import numpy as np
import torch


def set_seed(seed: int = 0):
    """
    Set the random seed.
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
