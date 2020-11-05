import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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
