import argparse
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import cv2
import numpy as np
import matplotlib.pyplot as plt
from piqa.ssim import SSIM
from piqa.tv import TV
from piqa.haarpsi import HaarPSI

from models import ConvAE, LinAE, ConvLinAE
from data import ReconDataModule, gather_NIND
from utils import plot_grad_flow, plot_samples


dev = 'cuda'
class AEDenoise(pl.LightningModule):
    def __init__(self, model, channels=3, multi_part=False): # 128, 20, 20
        super().__init__()
        self.model = model
        self.tv = TV()
        self.cri = SSIM()
        self.inv = True
        self.multi_part = multi_part

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # use l2 regularization
        optim = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim)
        return {'optimizer': optim, 'lr_scheduler': {'scheduler': sched, 'monitor': 'train loss'}}

    def training_step(self, batch, idx):
        x, y = batch
        y_h = self.forward(x)
        loss = self.calc_loss(y_h, y)
        # loss = self.cri(y_h, y)
        self.log('train loss', loss)
        if (idx+1) % 50 == 0:
            print(y.min().item(), y.max().item(), y_h.min().item(), y_h.max().item())
            tb = self.logger.experiment
            # tb.add_figure('grad',
            #             plot_grad_flow(self.model.named_parameters()),
            #             idx)
            tb.add_figure('ims', 
                        plot_samples(y_h, y, x), 
                        idx)
        return loss 
    
    def validation_step(self, batch, idx):
        x, y = batch
        y_h = self.forward(x)
        loss = self.calc_loss(y_h, y)
        # loss = self.cri(y_h, y)
        self.log('val loss', loss)
        return loss        
    
    # def validation_epoch_end(self, outputs):
    #     tb = self.logger.experiment
    #     # custom img
    #     noisy = 'chunked/NIND_Leonidas_ISO3200_16.png'
    #     clean = 'chunked/NIND_Leonidas_ISO200_16.png'
    #     # noisy = 'test_ims/train/train/114.png'
    #     # clean = 'test_ims/train_cleaned/train_cleaned/114.png'
    #     x = cv2.imread(noisy)
    #     xi = torch.FloatTensor(x).permute(2, 0, 1).unsqueeze(0).to(dev)
    #     y = cv2.imread(clean)
    #     y_h, mu, lv = self.forward(xi)
    #     y_h = y_h.squeeze(0).cpu().permute(1, 2, 0).numpy()
    #     tb.add_figure('ims', 
    #                   self.plot_samples(y_h, y, x), 
    #                   self.current_epoch)

    def calc_loss(self, y_h, y):
        inter = 1 - self.cri(y_h, y) if self.inv else self.cri(y_h, y)
        intra = self.tv(y_h) if self.multi_part else torch.zeros(inter.shape)
        intra = intra.type_as(inter)
        return 0.75 * inter + 0.25 * intra
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', default=10, type=int)
    parser.add_argument('-m', default=0, type=int)
    parser.add_argument('-c', default=3, type=int)
    parser.add_argument('-b', default=16, type=int)
    parser.add_argument('-l', default=False, type=bool)
    args = parser.parse_args()

    models = {
        0: ConvAE,
        1: LinAE,
        2: ConvLinAE
    }

    data = gather_NIND()
    print(f'Total Data size = {data.shape[0]}')

    tb = pl_loggers.TensorBoardLogger(save_dir='models/')
    datamodule = ReconDataModule(data, args.b)
    model = models[args.m](args.c, data_size=(3, 256, 256), latent_size=64)
    trainer = pl.Trainer(max_epochs=args.e, 
                         accelerator='gpu', devices=1, 
                         default_root_dir='models/',
                         logger=tb)
    trainer.fit(AEDenoise(model, channels=args.c, multi_part=args.l), datamodule=datamodule)