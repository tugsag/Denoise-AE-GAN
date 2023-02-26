import os
import argparse
from typing import Callable
import torch 
from torch import nn
from torchvision import transforms as tf
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dev = 'cuda'

class ReconDataset(torch.utils.data.Dataset):
    def __init__(self, npaths, cpaths) -> None:
        super().__init__()
        self.npaths = npaths
        self.cpaths = cpaths
        # self.ntrans = tf.Compose([
        #     tf.ToTensor(),
        #     tf.Normalize([45.92765673, 47.25423965, 64.29439433], [65.67457629, 65.90577327, 72.501963])
        # ])
        # self.ntrans = tf.Compose([
        #     tf.ToTensor(),
        #     tf.Grayscale(1),
        #     tf.Normalize([52.197041223914574], [66.04623369013014])
        # ])
        # self.ctrans = tf.Compose([
        #     tf.ToTensor(),
        #     tf.Normalize([41.35493992, 42.70296182, 57.99313621], [63.38838543, 63.60334709, 71.05605002])
        # ])
        # self.ctrans = tf.Compose([
        #     tf.ToTensor(),
        #     tf.Grayscale(1),
        #     tf.Normalize([46.82508484848485], [64.6721845622476])
        # ])

    def __len__(self):
        return len(self.npaths)
    
    def __getitem__(self, idx):
        # f = tf.Compose([
        #     tf.ToTensor(),
        #     tf.Grayscale(1)
        # ])
        cpath = self.cpaths[idx]
        npath = self.npaths[idx]
        cim = cv2.imread(cpath)/255.
        nim = cv2.imread(npath)/255.
        # cim = f(cim.astype(np.float32))
        # nim = f(nim.astype(np.float32))
        # cim = self.ctrans(cim)
        # nim = self.ntrans(nim)
        return torch.FloatTensor(nim).permute(2, 0, 1), torch.FloatTensor(cim).permute(2, 0, 1)
        # return nim, cim
    
class ReconDataModule(pl.LightningDataModule):
    def __init__(self, npaths, cpaths, batch_size) -> None:
        super().__init__()
        self.train, self.test = train_test_split(list(zip(npaths, cpaths)), 
                                                 test_size=0.2,
                                                 random_state=42,
                                                 shuffle=True)
        self.batch_size = batch_size

    def train_dataloader(self):
        n, c = zip(*self.train)
        assert all([i == j.replace('clean', 'noisy') for i, j in zip(n, c, strict=True)]), 'Mismatches!'
        return torch.utils.data.DataLoader(ReconDataset(n, c), 
                                           batch_size=self.batch_size,
                                           num_workers=8)
    
    def val_dataloader(self):
        n, c = zip(*self.test)
        assert all([i == j.replace('clean', 'noisy') for i, j in zip(n, c, strict=True)]), 'Mismatches!'
        return torch.utils.data.DataLoader(ReconDataset(n, c), 
                                           batch_size=self.batch_size,
                                           num_workers=8)
    

class ReconModel(pl.LightningModule):
    def __init__(self, channels=3, in_ch=(128, 20, 20)):
        super().__init__()
        self.in_ch = in_ch
        # enc
        self.conv1 = nn.Conv2d(channels, 128, 5, 2)
        self.conv2 = nn.Conv2d(128, 512, 5, 2)
        self.bnorm = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 256, 5, 2)
        self.conv4 = nn.Conv2d(256, 128, 3, 1)
        # lin
        self.lin1 = nn.Linear(np.prod(in_ch), 64)
        self.lin2 = nn.Linear(64, np.prod(in_ch))
        # dec
        self.deconv1 = nn.ConvTranspose2d(128, 256, 3, 1)
        self.deconv2 = nn.ConvTranspose2d(256, 512, 5, 2)
        self.deconv3 = nn.ConvTranspose2d(512, 128, 5, 2)
        self.deconv4 = nn.ConvTranspose2d(128, channels, 7, 2, output_padding=1)

    def sample(self, args):
        mu, lv = args
        if mu.min() == 0.:
            mu = mu + 1e-10
        if lv.min() == 0.:
            lv = lv + 1e-10
        std = torch.exp(lv/2)
        try:
            q = torch.distributions.Normal(mu, std)
        except ValueError as e:
            print(mu.min(), lv.min())
            raise ValueError
        z = q.rsample()
        return z

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optim
    
    def forward(self, x):
        batch = x.shape[0]
        # enc
        o = self.conv1(x)
        o = nn.functional.leaky_relu(self.conv2(o))
        o = self.bnorm(o)
        o = self.conv3(o)
        o = self.conv4(o)
        # lin/bottleneck ops
        e = self.lin1(torch.flatten(o, start_dim=1))
        lv = self.lin1(torch.flatten(o, start_dim=1))
        z = self.sample([e, lv])
        o = self.lin2(z)
        # dec
        o = self.deconv1(o.reshape(batch, *self.in_ch))
        o = nn.functional.leaky_relu(self.deconv2(o))
        o = self.bnorm(o)
        o = self.deconv3(o)
        o = self.deconv4(o)
        return z, e, lv, nn.functional.sigmoid(o)

    def training_step(self, batch, idx):
        x, y = batch
        z, e, lv, y_h = self.forward(x)
        # loss = nn.functional.mse_loss(y_h, y)
        loss = self.calc_loss(z, lv, y_h, y)
        self.log('train_MSE', loss)
        return loss 
    
    def validation_step(self, batch, idx):
        tb = self.logger.experiment
        x, y = batch
        z, e, lv, y_h = self.forward(x)
        # loss = nn.functional.mse_loss(y_h, y)
        loss = self.calc_loss(z, lv, y_h, y)
        self.log('val_MSE', loss)
        tb.add_figure('ims', self.plot_samples(y_h[-10].detach().cpu().permute(1, 2, 0).numpy(), y[-10].cpu().permute(1, 2, 0).numpy()), self.current_epoch)
        return loss
    
    def calc_loss(self, z, lv, pred, target):
        mse = nn.functional.mse_loss(pred, target)
        log_sigma_opt = 0.5 * mse.log()
        r_loss = 0.5 * torch.pow((target - pred) / log_sigma_opt.exp(), 2) + log_sigma_opt
        r_loss = r_loss.sum()
        kl_loss = self._compute_kl_loss(z, lv)
        return r_loss + kl_loss

    def _compute_kl_loss(self, z, lv): 
        return -0.5 * torch.sum(1 + lv - z.pow(2) - lv.exp())
    
    def plot_samples(self, out, target):
        out = np.clip(out, 0., 1.).astype(np.float32)
        target = np.clip(target, 0., 1.).astype(np.float32)
        f, ax = plt.subplots(1, 2, figsize=(10,5))
        ax[0].imshow(out)
        ax[0].set_title('model')
        ax[1].imshow(target)
        ax[1].set_title('target')

        return f

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', default=10, type=int)
    parser.add_argument('-c', default=3, type=int)
    parser.add_argument('-b', default=16, type=int)
    args = parser.parse_args()

    noise_paths = [f'noisy/{i}' for i in os.listdir('noisy')]
    clean_paths = [f'clean/{i}' for i in os.listdir('clean')]
    tb = pl_loggers.TensorBoardLogger(save_dir='models/')
    datamodule = ReconDataModule(noise_paths, clean_paths, args.b)
    trainer = pl.Trainer(max_epochs=args.e, 
                         accelerator='gpu', devices=1, 
                         default_root_dir='models/',
                         logger=tb)
    trainer.fit(ReconModel(channels=args.c), datamodule=datamodule)