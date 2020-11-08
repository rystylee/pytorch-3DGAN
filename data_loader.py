import glob

import numpy as np
import torch
import torch.utils.data as data

from utils import get_voxel_from_mat


def endless_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def load_dataloader(data_root, dataset_name, dataset_type, batch_size, dim_voxel):
    if dataset_type == 'raw':
        dataset = RawVolumeTextureDataset(
            data_root=data_root,
            dataset_name=dataset_name,
            dim_voxel=dim_voxel
            )
    elif dataset_type == 'shapenet':
        dataset = ShapeNetDataset(
            data_root=data_root,
            dataset_name=dataset_name,
            dim_voxel=dim_voxel
            )
    else:
        raise NotImplementedError()

    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    print(f'Total number : {len(dataset)}')
    return endless_dataloader(dataloader)


class RawVolumeTextureDataset(data.Dataset):
    def __init__(self, data_root, dataset_name, dim_voxel):
        self.dim_voxel = dim_voxel
        self.paths = glob.glob('{}/{}/train/*.bin'.format(data_root, dataset_name))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with open(self.paths[index], mode='rb') as f:
            data = np.fromfile(f, np.float32)
        return torch.from_numpy(data).view(self.dim_voxel, self.dim_voxel, self.dim_voxel)


class ShapeNetDataset(data.Dataset):
    def __init__(self, data_root, dataset_name, dim_voxel):
        self.dim_voxel = dim_voxel
        self.paths = glob.glob('{}/{}/30/train/*.mat'.format(data_root, dataset_name))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with open(self.paths[index], mode='rb') as f:
            data = np.asarray(get_voxel_from_mat(f, self.dim_voxel), np.float32)
        return torch.from_numpy(data).view(self.dim_voxel, self.dim_voxel, self.dim_voxel)
