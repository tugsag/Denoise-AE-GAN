import os
import argparse
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from piqa.ssim import MS_SSIM
from piqa.haarpsi import HaarPSI

from models import ConvAE, LinAE
from data import ReconDataModule, gather_NIND


dev = 'cuda'
class AEDenoise(pl.LightningModule):
    def __init__(self, model, channels=3): # 128, 20, 20
        super().__init__()
        self.model = model
        # self.cri = MS_SSIM(n_channels=channels, window_size=7, reduction='mean', sigma=1.5)
        self.cri = HaarPSI()
        # self.cri = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # use l2 regularization
        # TODO: add lr scheduler
        optim = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
        return optim

    def training_step(self, batch, idx):
        x, y = batch
        y_h, mu, lv = self.forward(x)
        # loss = self.calc_loss(y_h, y, mu, lv)
        loss = 1 - self.cri(y_h, y)
        self.log('train loss', loss)
        return loss 
    
    def validation_step(self, batch, idx):
        x, y = batch
        y_h, mu, lv = self.forward(x)
        # loss = self.calc_loss(y_h, y, mu, lv)
        loss = 1 - self.cri(y_h, y)
        self.log('val loss', loss)
        return loss
    
    def validation_epoch_end(self, outputs):
        tb = self.logger.experiment
        # custom img
        noisy = 'chunked/NIND_Leonidas_ISO3200_16.png'
        clean = 'chunked/NIND_Leonidas_ISO200_16.png'
        # noisy = 'test_ims/train/train/114.png'
        # clean = 'test_ims/train_cleaned/train_cleaned/114.png'
        x = cv2.imread(noisy)
        xi = torch.FloatTensor(x).permute(2, 0, 1).unsqueeze(0).to(dev)
        y = cv2.imread(clean)
        y_h, mu, lv = self.forward(xi)
        y_h = y_h.squeeze(0).cpu().permute(1, 2, 0).numpy()
        tb.add_figure('ims', 
                      self.plot_samples(y_h, y, x), 
                      self.current_epoch)

    def calc_loss(self, y_h, y, mu, lv):
        BCE = nn.functional.binary_cross_entropy(y_h, y, reduction='mean')
        KLD = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
        return BCE + KLD
    
    def plot_samples(self, out, target, original):
        f, ax = plt.subplots(1, 3, figsize=(15,5))
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        ax[0].imshow(out)
        ax[0].set_title('model')
        ax[1].imshow(np.clip(target/255., 0., 1.))
        ax[1].set_title('target')
        ax[2].imshow(np.clip(original/255.-out, 0., 1.))
        ax[2].set_title('diff')

        return f

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', default=10, type=int)
    parser.add_argument('-m', default=0, type=int)
    parser.add_argument('-c', default=3, type=int)
    parser.add_argument('-b', default=16, type=int)
    args = parser.parse_args()

    models = {
        0: ConvAE,
        1: LinAE
    }

    data = gather_NIND()
    print(f'Total Data size = {data.shape[0]}')

    tb = pl_loggers.TensorBoardLogger(save_dir='models/')
    datamodule = ReconDataModule(data, args.b)
    model = models[args.m](args.c, data_size=(3, 200, 200), latent_size=64)
    trainer = pl.Trainer(max_epochs=args.e, 
                         accelerator='gpu', devices=1, 
                         default_root_dir='models/',
                         logger=tb)
    trainer.fit(AEDenoise(model, channels=args.c), datamodule=datamodule)