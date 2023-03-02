import os
import argparse
import torch
from torch import nn
import torchvision
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from piqa.ssim import MS_SSIM, SSIM

from data import ReconDataModule
from models import Generator, Discrim


dev = 'cuda'
class GANDenoise(pl.LightningModule):
    def __init__(self, generator, discriminator) -> None:
        super().__init__()
        self.gen = generator
        self.dis = discriminator
        self.cri = nn.BCELoss()

    def forward(self, x):
        return self.gen(x)
    
    def adverse_loss(self, y_h, y):
        return self.cri(y_h, y)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        # train gen
        if optimizer_idx == 0:
            # generate images
            self.generated_imgs = self.forward(x)

            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image("generated_images", grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(x.size(0), 1)
            valid = valid.type_as(x)

            # adversarial loss is binary cross-entropy
            gen_loss = self.adverse_loss(self.dis(self.forward(z)), valid)
            self.log("gen_loss", gen_loss, prog_bar=True)
            return gen_loss
        
        # train dis
        if optimizer_idx == 1:
            # real
            valid = torch.ones(x.size(0), 1)
            valid = valid.type_as(y)
            real_loss = self.adverse_loss(self.dis(y), valid)

            # fake
            fake = torch.zeros(x.size(0), 1)
            fake = fake.type_as(y)
            fake_loss = self.adverse_loss(self.dis(self.forward(z).detach()), fake)

            dis_loss = (real_loss + fake_loss) / 2
            self.log("dis_loss", dis_loss, prog_bar=True)
            return dis_loss
        
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.gen.parameters())
        opt_d = torch.optim.Adam(self.dis.parameters())
        return [opt_g, opt_d], []
    
    # def on_validation_epoch_end(self) -> None:
    #     z = torch.randn(8, self.latent_dim).type_as(self.gen.model[0].weight)
    #     sample_ims = self.forward(z)
    #     grid = torchvision.utils.make_grid(sample_ims)
    #     self.logger.experiment.add_image('generated images', grid, self.current_epoch)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', default=10, type=int)
    parser.add_argument('-c', default=3, type=int)
    parser.add_argument('-b', default=16, type=int)
    args = parser.parse_args()

    noise_paths = [f'noisy/{i}' for i in os.listdir('noisy')]
    clean_paths = [f'clean/{i}' for i in os.listdir('clean')]
    # noise_paths = [os.path.join('test_ims/train/train/', i) for i in os.listdir('test_ims/train/train/')]
    # clean_paths = [os.path.join('test_ims/train_cleaned/train_cleaned/', i) for i in os.listdir('test_ims/train_cleaned/train_cleaned/')]
    tb = pl_loggers.TensorBoardLogger(save_dir='gans/')
    datamodule = ReconDataModule(noise_paths, clean_paths, args.b)
    gen = Generator((3, 200, 200), 100).to(dev)
    dis = Discrim(args.c).to(dev)
    trainer = pl.Trainer(max_epochs=args.e, 
                         accelerator='gpu', devices=1, 
                         default_root_dir='models/',
                         logger=tb)
    trainer.fit(GANDenoise(gen, dis), datamodule=datamodule)

