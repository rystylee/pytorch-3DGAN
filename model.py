import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, dim_z, ch, out_ch, bias):
        super(Generator, self).__init__()
        self.dim_z = dim_z
        self.ch = ch
        self.out_ch = out_ch
        self.bias = bias

        self.layer1 = self._conv_layer(self.dim_z, self.ch * 8, kernel_size=4, stride=2, padding=(0, 0, 0), bias=self.bias)
        self.layer2 = self._conv_layer(self.ch * 8, self.ch * 4, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)
        self.layer3 = self._conv_layer(self.ch * 4, self.ch * 2, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)
        self.layer4 = self._conv_layer(self.ch * 2, self.ch * 1, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)
        self.layer5 = self._final_layer(self.ch * 1, self.out_ch, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)

    def _conv_layer(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True)
        )
        return layer

    def _final_layer(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.Sigmoid()
            # nn.Tanh()
        )
        return layer

    def forward(self, x):
        out = x.view(-1, self.dim_z, 1, 1, 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = torch.squeeze(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, ch, out_ch, bias, dim_voxel):
        super(Discriminator, self).__init__()
        self.ch = ch
        self.out_ch = out_ch
        self.bias = bias
        # self.dim_voxel = dim_voxel

        self.layer1 = self._conv_layer(1, self.ch, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)
        self.layer2 = self._conv_layer(self.ch * 1, self.ch * 2, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)
        self.layer3 = self._conv_layer(self.ch * 2, self.ch * 4, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)
        self.layer4 = self._conv_layer(self.ch * 4, self.ch * 8, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)
        self.layer5 = self._final_layer(self.ch * 8, self.out_ch, kernel_size=4, stride=2, padding=(0, 0, 0), bias=self.bias)

    def _conv_layer(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return layer

    def _final_layer(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.Sigmoid()
        )
        return layer

    def forward(self, x):
        out = x.view(-1, 1, self.ch, self.ch, self.ch)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(-1, 1)
        return out
