import numpy as np
import torch
from torch import nn



class ConvL(nn.Module):
    def __init__(self, inch, outch, kernel, stride=1, pad=0, batch_norm=False) -> None:
        super().__init__()
        self.batch = nn.BatchNorm2d(outch) if batch_norm else None
        self.conv = nn.Conv2d(inch, outch, kernel, stride, padding=pad)

    def forward(self, x):
        x = nn.functional.relu(self.conv(x))
        if self.batch is not None:
            x = self.batch(x)
        return x
    
class DeconvL(nn.Module):
    def __init__(self, inch, outch, kernel, stride=1, output_padding=0) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose2d(inch, outch, kernel, stride, output_padding=output_padding)

    def forward(self, x):
        x = nn.functional.relu(self.deconv(x))
        return x
    
class LinL(nn.Module):
    def __init__(self, in_feat, out_feat, batch_norm=False) -> None:
        super().__init__()
        self.lin = nn.Linear(in_feat, out_feat)
        self.batch = nn.BatchNorm1d(out_feat) if batch_norm else None

    def forward(self, x):
        x = self.lin(x)
        if self.batch is not None:
            x = self.batch(x)
        return x
    
class GeneratorConv(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.c1 = ConvL(channels, 32, 5)
        self.c2 = ConvL(32, 64, 3, batch_norm=True)
        self.c3 = ConvL(64, 64, 3)
        self.c4 = ConvL(64, 128, 3, batch_norm=True)
        self.dc1 = DeconvL(128, 64, 3)
        self.dc2 = DeconvL(64, 64, 3)
        self.dc3 = DeconvL(64, 32, 3)
        self.dc4 = DeconvL(32, channels, 5)
        self.seq = nn.Sequential(
            self.c1, self.c2, self.c3, self.c4, 
            self.dc1, self.dc2, self.dc3, self.dc4, nn.Sigmoid()
        )

    def forward(self, x):
        return self.seq(x)

class GeneratorLin(nn.Module):
    def __init__(self, im_shape, latent_dim) -> None:
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
        self.c1 = ConvL(channels, 32, 5)
        self.c2 = ConvL(32, 64, 3)
        self.c3 = ConvL(64, 64, 3)
        self.c4 = ConvL(64, 128, 3)
        self.c5 = ConvL(128, 32, 3)
        self.lin1 = nn.Linear(32*188*188, 256)
        self.linout = nn.Linear(256, 1)
        # left manual declarations for easier debugging
        self.seq = nn.Sequential(
            self.c1, self.c2, self.c3, self.c4, self.c5
        )

    def forward(self, x):
        x = self.seq(x)
        x = self.lin1(torch.flatten(x, start_dim=1))
        return nn.functional.sigmoid(self.linout(x))

###################################

class ConvAE(nn.Module):
    def __init__(self, channels, **kws) -> None:
        super().__init__()
        self.c1 = ConvL(channels, 32, 15)
        self.c2 = ConvL(32, 64, 7, batch_norm=True)
        self.c3 = ConvL(64, 64, 5, batch_norm=True)
        self.c4 = ConvL(64, 128, 3, batch_norm=True)
        self.c5 = ConvL(128, 64, 2, batch_norm=True)
        self.dc1 = DeconvL(64, 128, 2)
        self.dc2 = DeconvL(128, 64, 3)
        self.dc3 = DeconvL(64, 64, 5)
        self.dc4 = DeconvL(64, 32, 7)
        self.dc5 = DeconvL(32, channels, 15)
        self.enc = nn.Sequential(
            self.c1, self.c2, self.c3, self.c4, self.c5
        )
        self.dec = nn.Sequential(
            self.dc1, self.dc2, self.dc3, self.dc4, self.dc5, nn.Sigmoid()
        )

    def forward(self, x):
        e = self.enc(x)
        return self.dec(e), e, 0

# self.encoder = nn.Sequential(
#             nn.Conv2d(channels, 32, 3, 1),
#             nn.LeakyReLU(),
#             nn.Conv2d(32, 64, 3, 1),
#             nn.Dropout(),
#             nn.Conv2d(64, 128, 3, 1),
#             nn.BatchNorm2d(128),
#             nn.Conv2d(128, 32, 3, 2),
#             nn.LeakyReLU(),
#             nn.Conv2d(32, 16, 5, 2)
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(16, 32, 5, 2),
#             nn.LeakyReLU(),
#             nn.ConvTranspose2d(32, 128, 5, 2, output_padding=1),
#             nn.Dropout(),
#             nn.ConvTranspose2d(128, 64, 3, 1),
#             nn.BatchNorm2d(64),
#             nn.ConvTranspose2d(64, 32, 3, 1),
#             nn.LeakyReLU(),
#             nn.ConvTranspose2d(32, 3, channels, 1),
#             nn.Sigmoid()
#         )
    
class LinAE(nn.Module):
    def __init__(self, channels, **kws) -> None:
        super().__init__()
        data_size = kws['data_size']
        latent_size = kws['latent_size']
        self.encoder = nn.Sequential(
            nn.Linear(np.prod(data_size), 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
            # nn.ReLU(),
            # nn.Linear(256, latent_size)
        )
        self.mu = nn.Linear(256, latent_size)
        self.lv = nn.Linear(256, latent_size)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, np.prod(data_size)),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, lv):
        std = torch.exp(.5*lv)
        eps = torch.rand_like(std)
        return mu + eps*std

    def forward(self, x):
        # x is full shaped, must reshape
        k = torch.flatten(x, start_dim=1)
        e = self.encoder(k)
        mu, lv = self.mu(e), self.lv(e)
        z = self.reparameterize(mu, lv)
        r = self.decoder(z)
        return r.reshape(x.shape), mu, lv
        