import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def sample_z(batch_size, dim_z, dis_z, device):
    if dis_z == 'norm':
        return torch.Tensor(batch_size, dim_z).normal_(0, 0.33).to(device)
    else:
        raise NotImplementedError()


def interpolate(a, b, t):
    return a + (b - a) * t


def save_voxel(samples, save_name, num_samples=8, thresh=0.1):
    samples = samples[:num_samples].__ge__(thresh)
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.set_aspect('equal')

    plt.savefig(save_name, bbox_inches='tight')
    plt.close()


def save_single_voxel(sample, save_name, thresh=0.1):
    sample = sample.__ge__(thresh)
    fig = plt.figure(figsize=(16, 16))

    x, y, z = sample.nonzero()
    ax = fig.gca(projection='3d')
    ax.scatter(x, y, z, zdir='z', c='red')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # ax.set_aspect('equal')

    plt.savefig(save_name, bbox_inches='tight')
    plt.close()
