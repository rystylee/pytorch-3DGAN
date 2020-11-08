import os
from tqdm import tqdm

import torch

from model import Generator
from utils import sample_z, interpolate, save_single_voxel


class Tester(object):
    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.dim_z = config.dim_z
        self.dis_z = config.dis_z

        self.generator = Generator(dim_z=config.dim_z, ch=config.ch_g, out_ch=1, bias=config.bias).to(self.device)
        self._load_models(config.checkpoint_path)
        self.generator.eval()

    def test(self):
        src_z = sample_z(1, self.dim_z, self.dis_z, self.device)
        dst_z = sample_z(1, self.dim_z, self.dis_z, self.device)

        counter = 0
        for i in tqdm(range(1200)):
            with torch.no_grad():
                z = interpolate(src_z, dst_z, counter)

                out = self.generator(z)
                out = out.detach().cpu().numpy()

                binary_path = os.path.join(self.config.binary_dir, f'{i:05}.bin')
                out.tofile(binary_path)

                img_path = os.path.join(self.config.img_dir, f'{i:05}.jpg')
                save_single_voxel(out, img_path, 0.05)

                counter += 0.005
                if counter > 1.0:
                    counter = 0
                    src_z = dst_z
                    dst_z = sample_z(1, self.dim_z, self.dis_z, self.device)

    def _load_models(self, model_state_path):
        checkpoint = torch.load(model_state_path, map_location=torch.device('cpu'))
        self.generator.load_state_dict(checkpoint['generator'])
        print('Loaded pretrained models...\n')
