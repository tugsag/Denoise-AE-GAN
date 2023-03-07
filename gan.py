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
    def __init__(self, generator: torch.nn.Module, discriminator: torch.nn.Module, content_: torch.nn.Module = None) -> None:
        super().__init__()
        self.gen = generator
        self.dis = discriminator
        self.content_ = content_
        self.tv = TV()
        self.dis_loss = torch.nn.BCELoss()

    def forward(self, x):
        return self.gen(x)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch

        # generate images
        generated_imgs = self.forward(x)
        disg_outs = self.dis(generated_imgs)

        # train gen
        if optimizer_idx == 0:
            # log sampled images
            sample_imgs = generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image("generated_images", grid, 0)

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
    
    def get_feature_loss(self, real_im, gen_out):
        ftt = self.content_(real_im)
        gtt = self.content_(gen_out)
        feature_count = ftt.shape[1]
        floss = torch.nn.functional.mse_loss(ftt, gtt) / feature_count
        return floss
    
    def get_smooth_loss(self, gen_out):
        # b, h, w = gen_out.shape[0], gen_out.shape[2], gen_out.shape[3]
        # hn = gen_out[:b, :, :h, :w-1]
        # hr = gen_out[:b, :, :h, 1:w-1]
        # vn = gen_out[:b, :, :h-1, :w]
        # vr = gen_out[:b, :, 1:h-1, :w]
        # return torch.nn.functional.mse_loss(hn, hr) + torch.nn.functional.mse_loss(vn, vr)
        return self.tv(gen_out)

    def calculate_g_loss(self, gen_out, disg_loss, real_im):
        pixel_loss = torch.nn.functional.mse_loss(gen_out, real_im)
        smooth_loss = self.get_smooth_loss(gen_out)
        feature_loss = self.get_feature_loss(real_im, gen_out)
        gloss = 0.5 * disg_loss + pixel_loss + 0.0001*smooth_loss + feature_loss
        return gloss
        
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
    vgg = torchvision.models.vgg16(pretrained=True)
    vgg_cont = torch.nn.Sequential(*list(vgg.children())[:-2])
    trainer = pl.Trainer(max_epochs=args.e, 
                         accelerator='gpu', devices=1, 
                         default_root_dir='models/',
                         logger=tb)
    trainer.fit(GANDenoise(gen, dis, vgg_cont), datamodule=datamodule)

