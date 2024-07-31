import os
import random
import numpy as np
import torch


def seed_everything(yaml_args, command_args):
    os.environ["PYTHONHASHSEED"] = str(yaml_args.seed)
    random.seed(yaml_args.seed)
    np.random.seed(yaml_args.seed)
    torch.manual_seed(yaml_args.seed)
    if command_args.n_gpu:
        torch.cuda.manual_seed_all(yaml_args.seed)


class EarlyStopping(object):
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
