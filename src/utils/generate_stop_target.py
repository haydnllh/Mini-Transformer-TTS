import torch

def generate_stop_target(padding_mask):
    stop_target = torch.zeros_like(padding_mask, dtype=torch.float32)

    lengths = torch.sum((~padding_mask).float(), dim=1)

    for i, l in enumerate(lengths):
        if l > 0:
            stop_target[i, int(l.item()) - 1] = 1
        else:
            stop_target[i, 0] = 1

    return stop_target