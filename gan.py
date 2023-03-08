import argparse
from itertools import chain
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np
import matplotlib.pyplot as plt
from piqa.ssim import SSIM
from piqa.tv import TV

from data import ReconDataModule, gather_NIND
from models import GeneratorConv, Discrim


dev = 'cuda'
class GANDenoise(pl.LightningModule):
    def __init__(self, generator: torch.nn.Module, discriminator: torch.nn.Module) -> None:
        super().__init__()
        self.gen = generator
        self.dis = discriminator
        self.tv = TV()
        self.dis_loss = torch.nn.BCELoss()
        self.con_loss = torch.nn.L1Loss()

    def forward(self, x):
        return self.gen(x)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch

        # generate image and discriminate
        generated_imgs = self.forward(x)
        disg_outs = self.dis(generated_imgs)

        # train gen
        if optimizer_idx == 0:
            # log sampled images
            if (batch_idx + 1) % 30 == 0:
                sample_imgs = torch.cat([generated_imgs[:3], y[:3]])
                grid = torchvision.utils.make_grid(sample_imgs)
                self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            fake = torch.zeros(x.size(0), 1)
            fake = fake.type_as(x)

            # adversarial loss is binary cross-entropy
            disg_loss = self.dis_loss(disg_outs, fake)
            gen_loss = self.calculate_g_loss(generated_imgs, disg_loss, y)
            self.log("gen_loss", gen_loss, prog_bar=True)
            return gen_loss
        
        # train dis
        if optimizer_idx == 1:
            # real
            valid = torch.ones(x.size(0), 1)
            valid = valid.type_as(y)

            # fake
            fake = torch.zeros(x.size(0), 1)
            fake = fake.type_as(y)

            disr_outs = self.dis(y)
            dis_loss = self.calculate_d_loss(disg_outs, fake, disr_outs, valid)
            self.log("dis_loss", dis_loss, prog_bar=True)
            return dis_loss
        
    def calculate_d_loss(self, disg_out, fake, disr_out, valid):
        return (self.dis_loss(disg_out, fake) + self.dis_loss(disr_out, valid)) / 2
        # return -torch.mean(torch.log(disr_out) + torch.log(1-disg_out))
    
    def calculate_g_loss(self, gen_out, disg_loss, real_im):
        content = self.con_loss(gen_out, real_im)
        return 0.9 * content + 0.1 * disg_loss
        
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.gen.parameters())
        # d_comb_params = chain(self.disr.parameters(), self.disf.parameters())
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

    df = gather_NIND()
    tb = pl_loggers.TensorBoardLogger(save_dir='gans/')
    datamodule = ReconDataModule(df, args.b)
    gen = GeneratorConv(3, latent_dim=100).to(dev)
    dis = Discrim(args.c).to(dev)

    trainer = pl.Trainer(max_epochs=args.e, 
                         accelerator='gpu', devices=1, 
                         default_root_dir='models/',
                         logger=tb)
    trainer.fit(GANDenoise(gen, dis), datamodule=datamodule)

