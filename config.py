import time
import os
import json
import argparse


def get_config():
    parser = argparse.ArgumentParser()

    # mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # dataset
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--dataset_name', type=str, default='fluid')
    parser.add_argument('--dataset_type', type=str, default='raw', choices=['raw', 'shapenet'])
    parser.add_argument('--batch_size', type=int, default=32)

    # training
    parser.add_argument('--max_itr', type=int, default=450000)
    parser.add_argument('--lr_g', type=float, default=0.0025)
    parser.add_argument('--lr_d', type=float, default=0.00001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--checkpoint_path', type=str, default='')

    # model
    parser.add_argument('--bias', action='store_true')

    # generator
    parser.add_argument('--ch_g', type=int, default=64)
    parser.add_argument('--dim_z', type=int, default=200)
    parser.add_argument('--dis_z', type=str, default='norm')

    # discriminator
    parser.add_argument('--ch_d', type=int, default=64)
    parser.add_argument('--dim_voxel', type=int, default=64)
    parser.add_argument('--d_thresh', type=float, default=0.8)

    # misc
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--checkpoint_interval', type=int, default=100)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--sample_dir', type=str, default='samples')
    parser.add_argument('--sample_interval', type=int, default=100)
    parser.add_argument('--test_dir', type=str, default='results')

    # testing
    parser.add_argument('--config_path', type=str, default='')
    parser.add_argument('--img_dir', type=str, default='images')
    parser.add_argument('--binary_dir', type=str, default='binaries')

    args = parser.parse_args()

    time_str = time.strftime("%Y%m%d-%H%M%S")
    config_name = f'{time_str}-{args.dataset_name}-{args.dim_voxel}'

    runs_path = os.path.join('runs', config_name)
    args.log_dir = os.path.join(runs_path, args.log_dir)
    args.checkpoint_dir = os.path.join(runs_path, args.checkpoint_dir)
    args.sample_dir = os.path.join(runs_path, args.sample_dir)

    if args.mode == 'train':
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.sample_dir, exist_ok=True)
        with open(os.path.join(runs_path, 'config.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)

    return args
