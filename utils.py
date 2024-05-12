import os
import torch
import torch.nn as nn


def create_labels(n: int, r1: float, r2: float, device: torch.device = None):
    return torch.empty(n, 1, requires_grad=False, device=device).uniform_(r1, r2)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def create_custom_dir(directory_path: str, use_unique: bool = True) -> str:
    """Creates a new directory (directory_path).
    If path name already exist, it adds a numerical suffix to the directory name"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        return directory_path
    else:
        if use_unique:
            count = 1
            while True:
                new_directory_path = f"{directory_path}_{count}"
                if not os.path.exists(new_directory_path):
                    os.makedirs(new_directory_path)
                    return new_directory_path
                count += 1
        else:
            return directory_path
