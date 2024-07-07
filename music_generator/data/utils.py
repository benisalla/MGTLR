import numpy as np
import torch


def get_batch(data, seq_len, b_size):
    if len(data) <= seq_len:
        raise ValueError(f"Data length ({len(data)}) must be greater than seq_len ({seq_len}).")
    
    idx = np.random.choice(len(data)-seq_len-1, b_size)

    x = np.reshape([data[i:i+seq_len] for i in idx], [b_size, seq_len])
    y = np.reshape([data[i+1:i+seq_len+1] for i in idx], [b_size, seq_len])

    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)