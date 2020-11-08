import os

from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import Discriminator
from model import Generator
from losses import GANLoss
from data_loader import load_dataloader
from utils import sample_z, save_voxel


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.dataloader = load_dataloader(
            data_root=config.data_root,
            dataset_name=config.dataset_name,
            batch_size=config.batch_size,
            dim_voxel=config.dim_voxel
            )

        self.start_itr = 1
        self.dim_z = config.dim_z
        self.dis_z = config.dis_z
        self.d_thresh = config.d_thresh

        self.generator = Generator(dim_z=config.dim_z, ch=config.ch_g, out_ch=1, bias=config.bias).to(self.device)
        self.discriminator = Discriminator(ch=config.ch_d, out_ch=1, bias=config.bias, dim_voxel=config.dim_voxel).to(self.device)

        self.optim_g = optim.Adam(self.generator.parameters(), lr=config.lr_g, betas=(config.beta1, config.beta2))
        self.optim_d = optim.Adam(self.discriminator.parameters(), lr=config.lr_d, betas=(config.beta1, config.beta2))
        self.criterion = GANLoss()

        if not self.config.checkpoint_path == '':
            self._load_models(self.config.checkpoint_path)

        self.writer = SummaryWriter(log_dir=config.log_dir)

    def train(self):
        print('Start training!\n')
        with tqdm(total=self.config.max_itr + 1 - self.start_itr) as pbar:
            for n_itr in range(self.start_itr, self.config.max_itr + 1):
                pbar.set_description(f'iteration [{n_itr}]')

                img = next(self.dataloader)
                real_img = img.to(self.device)

                batch_size = len(real_img)

                # ------------------------------------------------
                # Train D
                # ------------------------------------------------
                z = sample_z(batch_size, self.dim_z, self.dis_z, self.device)

                with torch.no_grad():
                    fake_img = self.generator(z)

                d_real = self.discriminator(real_img)
                d_fake = self.discriminator(fake_img)

                loss_d_real = self.criterion(d_real, 'd_real')
                loss_d_fake = self.criterion(d_fake, 'd_fake')
                loss_d = loss_d_real + loss_d_fake

                acc_d_real = torch.ge(d_real.squeeze(), 0.5).float()
                acc_d_fake = torch.le(d_fake.squeeze(), 0.5).float()
                acc_d = torch.mean(torch.cat((acc_d_real, acc_d_fake), 0))

                if acc_d < self.d_thresh:
                    self.optim_d.zero_grad()
                    loss_d.backward()
                    self.optim_d.step()

                # ------------------------------------------------
                # Train G
                # ------------------------------------------------
                z = sample_z(batch_size, self.dim_z, self.dis_z, self.device)

                fake_img = self.generator(z)
                d_fake = self.discriminator(fake_img)

                loss_g = self.criterion(d_fake, 'g')

                self.optim_g.zero_grad()
                loss_g.backward()
                self.optim_g.step()

                # ------------------------------------------------
                #  Logging
                # ------------------------------------------------
                if n_itr % self.config.log_interval == 0:
                    tqdm.write('iteration: {}/{}, loss_g: {}, loss_d: {}, loss_d_real: {}, loss_d_fake: {}'.format(
                        n_itr, self.config.max_itr, loss_g.item(), loss_d.item(), loss_d_real.item(), loss_d_fake.item()
                        ))
                    self.writer.add_scalar('loss/loss_g', loss_g.item(), n_itr)
                    self.writer.add_scalar('loss/loss_d', loss_d.item(), n_itr)
                    self.writer.add_scalar('loss/loss_d_real', loss_d_real.item(), n_itr)
                    self.writer.add_scalar('loss/loss_d_fake', loss_d_fake.item(), n_itr)
                    self.writer.add_scalar('loss/acc_d', acc_d.item(), n_itr)

                # ------------------------------------------------
                #  Sampling
                # ------------------------------------------------
                if n_itr % self.config.sample_interval == 0:
                    img_path = os.path.join(self.config.sample_dir, f'fake_{n_itr}.jpg')
                    samples = fake_img.detach().cpu().numpy()
                    save_voxel(samples, img_path)

                if n_itr % self.config.sample_interval == 0:
                    img_path = os.path.join(self.config.sample_dir, f'real_{n_itr}.jpg')
                    samples = real_img.detach().cpu().numpy()
                    save_voxel(samples, img_path)

                # ------------------------------------------------
                #  Save model
                # ------------------------------------------------
                if n_itr % self.config.checkpoint_interval == 0:
                    self._save_models(n_itr)

                pbar.update()

        self.writer.close()

    def _save_models(self, n_itr):
        checkpoint_name = f'{self.config.dataset_name}-{self.config.dim_voxel}_model_ckpt_{n_itr}.pth'
        checkpoint_path = os.path.join(self.config.checkpoint_dir, checkpoint_name)
        torch.save({
            'n_itr': n_itr,
            'generator': self.generator.state_dict(),
            'optim_g': self.optim_g.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optim_d': self.optim_d.state_dict(),
        }, checkpoint_path)
        tqdm.write(f'Saved checkpoint: n_itr_{n_itr}')

    def _load_models(self, model_state_path):
        checkpoint = torch.load(model_state_path)
        self.start_itr = checkpoint['n_itr'] + 1
        self.generator.load_state_dict(checkpoint['generator'])
        self.optim_g.load_state_dict(checkpoint['optim_g'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optim_d.load_state_dict(checkpoint['optim_d'])
        print(f'start_itr: {self.start_itr}')
        print('Loaded pretrained models...\n')
