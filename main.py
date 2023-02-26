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
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from ssim_loss import MS_SSIM

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
        f = tf.Compose([
            tf.ToTensor(),
            tf.Resize((420, 540)), 
            tf.Grayscale(1)
        ])
        cpath = self.cpaths[idx]
        npath = self.npaths[idx]
        cim = cv2.imread(cpath)/255.
        nim = cv2.imread(npath)/255.
        cim = f(cim.astype(np.float32))
        nim = f(nim.astype(np.float32))
        # cim = self.ctrans(cim)
        # nim = self.ntrans(nim)
        # return torch.FloatTensor(nim).permute(2, 0, 1), torch.FloatTensor(cim).permute(2, 0, 1)
        return nim, cim
    
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
        assert all([j == i.replace('train', 'train_cleaned') for i, j in zip(n, c, strict=True)]), 'Mismatches!'
        return torch.utils.data.DataLoader(ReconDataset(n, c), 
                                           batch_size=self.batch_size,
                                           num_workers=8)
    
    def val_dataloader(self):
        n, c = zip(*self.test)
        assert all([j == i.replace('train', 'train_cleaned') for i, j in zip(n, c, strict=True)]), 'Mismatches!'
        return torch.utils.data.DataLoader(ReconDataset(n, c), 
                                           batch_size=self.batch_size,
                                           num_workers=8)
    

class ReconModel(pl.LightningModule):
    def __init__(self, channels=3, in_ch=(128, 47, 62)): # 128, 20, 20
        super().__init__()
        self.in_ch = in_ch
        # self.cri = nn.BCELoss()
        self.cri = MS_SSIM(data_range=1., channel=channels)
        # enc
        self.conv1 = nn.Conv2d(channels, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 256, 3, 1)
        self.bnorm = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 64, 3, 2)
        self.conv4 = nn.Conv2d(64, 32, 5, 2)
        # lin
        # self.lin1 = nn.Linear(np.prod(in_ch), 64)
        # self.lin2 = nn.Linear(64, np.prod(in_ch))
        self.do = nn.Dropout()
        # dec
        self.deconv1 = nn.ConvTranspose2d(32, 64, 5, 2)
        self.deconv2 = nn.ConvTranspose2d(64, 256, 3, 2, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 64, 3, 1) # kernel size of 5
        self.deconv4 = nn.ConvTranspose2d(64, channels, 3, 1)
        self.sig = nn.Sigmoid()

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optim
    
    def forward(self, x):
        batch = x.shape[0]
        # enc
        o = self.conv1(x)
        o = nn.functional.leaky_relu(self.conv2(o))
        o = self.bnorm(o)
        o = self.conv3(o)
        o = nn.functional.leaky_relu(self.conv4(o))
        # lin/bottleneck ops
        # e = self.lin1(torch.flatten(o, start_dim=1))
        # o = self.lin2(e)
        o = self.do(o)
        # dec
        # o = self.deconv1(o.reshape(batch, *self.in_ch))
        o = self.deconv1(o)
        o = nn.functional.leaky_relu(self.deconv2(o))
        o = self.bnorm(o)
        o = self.deconv3(o)
        o = nn.functional.leaky_relu(self.deconv4(o))
        return 0, self.sig(o)

    def training_step(self, batch, idx):
        x, y = batch
        e, y_h = self.forward(x)
        loss = 1 - self.cri(y_h, y)
        self.log('train loss', loss)
        return loss 
    
    def validation_step(self, batch, idx):
        x, y = batch
        e, y_h = self.forward(x)
        loss = 1 - self.cri(y_h, y)
        # loss = self.calc_loss(y_h, y)
        self.log('val loss', loss)
        return loss
    
    def validation_epoch_end(self, outputs):
        tb = self.logger.experiment
        # custom img
        # noisy = 'noisy/01_noisy_138.png'
        # clean = 'clean/01_clean_138.png'
        noisy = 'test_ims/train/train/114.png'
        clean = 'test_ims/train_cleaned/train_cleaned/114.png'
        noisy_im = cv2.imread(noisy)
        noisy_im = torch.FloatTensor(cv2.cvtColor(noisy_im, cv2.COLOR_BGR2GRAY)).unsqueeze(-1).permute(2, 0, 1).unsqueeze(0).to(dev)
        y = cv2.imread(clean)
        e, y_h = self.forward(noisy_im)
        print('pred stats: ', y_h.min(), y_h.max())
        tb.add_figure('ims', self.plot_samples(y_h.squeeze(0).cpu().permute(1, 2, 0).numpy(), y), self.current_epoch)
    
    def plot_samples(self, out, target):
        f, ax = plt.subplots(1, 2, figsize=(10,5))
        ax[0].imshow(np.clip(out*255, 0., 255.), cmap='gray')
        ax[0].set_title('model')
        ax[1].imshow(np.clip(target*255, 0., 255.), cmap='gray')
        ax[1].set_title('target')

        return f

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', default=10, type=int)
    parser.add_argument('-c', default=3, type=int)
    parser.add_argument('-b', default=16, type=int)
    args = parser.parse_args()

    # noise_paths = [f'noisy/{i}' for i in os.listdir('noisy')]
    # clean_paths = [f'clean/{i}' for i in os.listdir('clean')]
    noise_paths = [os.path.join('test_ims/train/train/', i) for i in os.listdir('test_ims/train/train/')]
    clean_paths = [os.path.join('test_ims/train_cleaned/train_cleaned/', i) for i in os.listdir('test_ims/train_cleaned/train_cleaned/')]
    tb = pl_loggers.TensorBoardLogger(save_dir='models/')
    datamodule = ReconDataModule(noise_paths, clean_paths, args.b)
    trainer = pl.Trainer(max_epochs=args.e, 
                         accelerator='gpu', devices=1, 
                         default_root_dir='models/',
                         logger=tb)
    trainer.fit(ReconModel(channels=args.c), datamodule=datamodule)