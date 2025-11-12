import random
import torch
import torch.backends.cudnn
import time
import numpy as np


def set_random(seed: int = int(time.time())):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def min_max_scale(x: [], half_OR_zero_center: bool = False) -> [float]:
    x1d = x.ravel()
    min_value = min(x1d)
    max_value = max(x1d)
    x_range = max_value - min_value

    if x_range == 0:
        return x

    if not half_OR_zero_center:
        numerator = x - min_value
    else:
        avg_value = sum(x1d) / len(x1d)
        numerator = x - avg_value
    x = numerator / x_range
    return x


def get_sequence_no(no: int) -> str:
    if no == 1:
        postfix = "st"
    elif no == 2:
        postfix = "nd"
    elif no == 3:
        postfix = "rd"
    else:
        postfix = "th"
    sequence_no = str(no) + postfix
    return sequence_no
