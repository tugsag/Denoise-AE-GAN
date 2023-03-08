import numpy as np
import torch
from torch import nn



class Residual(nn.Module):
    def __init__(self, inch, outch, kernel, stride=1, pad=1) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(inch, outch, kernel, stride, padding=pad)
        self.c2 = nn.Conv2d(outch, outch, kernel, stride, padding=pad)
        self.b1 = nn.BatchNorm2d(outch)
        self.b2 = nn.BatchNorm2d(outch)

    def forward(self, x):
        r = nn.functional.leaky_relu(self.b1(self.c1(x)))
        r = nn.functional.leaky_relu(self.b2(self.c2(x)))
        return x + r

class ConvL(nn.Module):
    def __init__(self, inch, outch, kernel, stride=1, pad=0, batch_norm=True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(inch, outch, kernel, stride, padding=pad)
        self.batch = nn.BatchNorm2d(outch) if batch_norm else None

    def forward(self, x):
        x = nn.functional.leaky_relu(self.conv(x))
        # x = self.conv(x)
        if self.batch is not None:
            x = self.batch(x)
        return x
    
class DeconvL(nn.Module):
    def __init__(self, inch, outch, kernel, stride=1, output_padding=0) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose2d(inch, outch, kernel, stride, output_padding=output_padding)

    def forward(self, x):
        # x = nn.functional.leaky_relu(self.deconv(x))
        x = self.deconv(x)
        return x
    
class LinL(nn.Module):
    def __init__(self, in_feat, out_feat, batch_norm=False) -> None:
        super().__init__()
        self.lin = nn.Linear(in_feat, out_feat)
        self.batch = nn.BatchNorm1d(out_feat) if batch_norm else None

    def forward(self, x):
        x = nn.functional.leaky_relu(self.lin(x))
        if self.batch is not None:
            x = self.batch(x)
        return x
    
##########################################
    
class GeneratorConv(nn.Module):
    def __init__(self, channels, **kws) -> None:
        super().__init__()
        self.c1 = ConvL(channels, 32, 7)
        self.c2 = ConvL(32, 64, 3, batch_norm=True)
        self.c3 = ConvL(64, 64, 3, batch_norm=True)
        self.r1 = Residual(64, 64, 3)
        self.r2 = Residual(64, 64, 3)
        self.r3 = Residual(64, 64, 3)
        self.dc1 = DeconvL(64, 64, 3)
        self.dc2 = DeconvL(64, 32, 5)
        self.dc3 = DeconvL(32, 32, 7)
        self.fc1 = ConvL(32, 3, 3, batch_norm=True)
        self.out = nn.Sigmoid()

        self.seq = nn.Sequential(
            self.c1, self.c2, self.c3,
            self.r1, self.r2, self.r3,
            self.dc1, self.dc2, self.dc3,
            self.fc1, self.out
        )

    def forward(self, x):
        return self.seq(x)
        
class GeneratorLin(nn.Module):
    def __init__(self, im_shape, latent_dim=100) -> None:
        super().__init__()
        self.im_shape = im_shape
        self.lin1 = LinL(latent_dim, 128)
        self.lin2 = LinL(128, 512, batch_norm=True)
        self.lin3 = LinL(512, 1024)
        self.lin4 = LinL(1024, 2048, batch_norm=True)
        self.lin5 = LinL(2048, np.prod(im_shape))
        self.seq = nn.Sequential(
            self.lin1, self.lin2, self.lin3, self.lin4, self.lin5, nn.Tanh()
        )

    def forward(self, x):
        x = self.seq(x)
        return x.view(x.size(0), *self.im_shape)
        
class Discrim(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.c1 = ConvL(channels, 32, 4, stride=2)
        self.c2 = ConvL(32, 64, 4, stride=2)
        self.c3 = ConvL(64, 64, 4, stride=2)
        self.r1 = Residual(64, 64, 3)
        self.r2 = Residual(64, 64, 3)
        self.c4 = ConvL(64, 128, 4)
        self.c5 = ConvL(128, 32, 4)
        self.linout = nn.Linear(32*42*42, 1)
        self.out = nn.Sigmoid()
        # left manual declarations for easier debugging
        self.seq = nn.Sequential(
            self.c1, self.c2, self.c3, self.r1, self.r2, self.c4, self.c5
        )

    def forward(self, x):
        x = self.seq(x)
        x = self.linout(torch.flatten(x, start_dim=1))
        return self.out(x)

###################################

class ConvAE(nn.Module):
    def __init__(self, channels, **kws) -> None:
        super().__init__()
        self.c1 = ConvL(channels, 32, 7)
        self.cstride1 = ConvL(32, 64, 5, stride=2)
        self.r1 = Residual(64, 64, 3)
        self.r2 = Residual(64, 64, 3)
        self.r3 = Residual(64, 64, 3)
        self.r4 = Residual(64, 64, 3)
        self.cstride2 = ConvL(64, 128, 3, stride=2)
        self.c5 = ConvL(128, 64, 3)

        self.dc1 = DeconvL(64, 128, 3)
        self.dcstride1 = DeconvL(128, 64, 3, stride=2)
        self.dcstride2 = DeconvL(64, 64, 3, stride=2, output_padding=1)
        self.dc4 = DeconvL(64, 32, 3)
        self.dc5 = DeconvL(32, channels, 7)

        self.enc = nn.Sequential(
            self.c1, self.cstride1,
            self.r1, self.r2, self.r3, self.r4, 
            self.cstride2, self.c5
        )
        self.dec = nn.Sequential(
            self.dc1, self.dcstride1, self.dcstride2, self.dc4, self.dc5, nn.Sigmoid()
        )

    def forward(self, x):
        e = self.enc(x)
        d = self.dec(e)
        return d
    
class ConvLinAE(nn.Module):
    def __init__(self, channels, **kws) -> None:
        super().__init__()
        self.c1 = ConvL(channels, 32, 5, batch_norm=True)
        self.c2 = ConvL(32, 64, 5, batch_norm=True)
        self.cstride1 = ConvL(64, 64, 3, stride=2, batch_norm=True)
        self.c3 = ConvL(64, 64, 3, batch_norm=True)
        self.c4 = ConvL(64, 128, 3, batch_norm=True)
        self.cstride2 = ConvL(128, 32, 3, stride=2, batch_norm=True)
        self.pool = nn.MaxPool2d(3, 2)

        self.lin1 = LinL(32*59*59, 512, batch_norm=True)
        self.lin2 = LinL(512, 128)
        self.lin3 = LinL(128, 512)
        self.lin4 = LinL(512, 32*59*59, batch_norm=True)

        self.dcstride0 = DeconvL(32, 32, 3, stride=2)
        self.dcstride1 = DeconvL(32, 128, 3, stride=2)
        # self.dc1 = DeconvL(32, 128, 3)
        self.dc2 = DeconvL(128, 64, 3)
        self.dc3 = DeconvL(64, 64, 5)
        self.dcstride2 = DeconvL(64, 64, 3, stride=2, output_padding=1)
        self.dc4 = DeconvL(64, 32, 5)
        self.dc5 = DeconvL(32, channels, 5)
        self.enc = nn.Sequential(
            self.c1, self.c2, self.cstride1, self.c3, self.c4, self.pool, self.cstride2
        )
        # self.enc2 = nn.Sequential(
        #     self.lin1, self.lin2, self.lin3, self.lin4
        # )
        self.dec = nn.Sequential(
            self.dcstride0, self.dcstride1, self.dc2, self.dc3, self.dcstride2, self.dc4, self.dc5, nn.Sigmoid()
        )

    def forward(self, x):
        e = self.enc(x)
        b = self.lin1(torch.flatten(e, start_dim=1))
        b = self.lin2(b)
        b = self.lin3(b)
        b = self.lin4(b)
        return self.dec(b.view(x.shape[0], 32, 59, 59))

    
class LinAE(nn.Module):
    def __init__(self, channels, **kws) -> None:
        super().__init__()
        data_size = kws['data_size']
        self.encoder = nn.Sequential(
            nn.Linear(np.prod(data_size), 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
            # nn.ReLU(),
            # nn.Linear(128, latent_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, np.prod(data_size)),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x is full shaped, must reshape
        k = torch.flatten(x, start_dim=1)
        e = self.encoder(k)
        out = self.decoder(e)
        return out.view(x.shape)
        