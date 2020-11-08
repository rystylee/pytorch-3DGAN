import torch

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io as io
import scipy.ndimage as nd
import numpy as np


def sample_z(batch_size, dim_z, dis_z, device):
    if dis_z == 'norm':
        return torch.Tensor(batch_size, dim_z).normal_(0, 0.33).to(device)
    else:
        raise NotImplementedError()


def interpolate(a, b, t):
    return a + (b - a) * t


def get_voxel_from_mat(path, dim_voxel):
    voxels = io.loadmat(path)['instance']  # 30x30x30
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))  # 32x32x32
    if dim_voxel == 64:
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)  # 64x64x64
    if dim_voxel == 128:
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)  # 128x128x128
    return voxels


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
