import torch
import torch.nn as nn


class GANLoss(object):
    def __init__(self):
        self.loss_fn = nn.BCELoss()

    def __call__(self, logits, loss_type, soft=True):
        assert loss_type in ['g', 'd_real', 'd_fake']
        batch_size = len(logits)
        device = logits.device
        if loss_type == 'g':
            label = torch.Tensor(batch_size, 1).uniform_(0.7, 1.2).to(device) if soft else torch.ones(batch_size, 1).to(device)
            return self.loss_fn(logits, label)
        elif loss_type == 'd_real':
            label = torch.Tensor(batch_size, 1).uniform_(0.7, 1.2).to(device) if soft else torch.ones(batch_size, 1).to(device)
            return self.loss_fn(logits, label)
        if loss_type == 'd_fake':
            label = torch.Tensor(batch_size, 1).uniform_(0.0, 0.3).to(device) if soft else torch.zeros(batch_size, 1).to(device)
            return self.loss_fn(logits, label)
