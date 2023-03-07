import glob
import os
import torch
from torchvision import transforms as tf
import pytorch_lightning as pl
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split




dev = 'cuda'
def gather_NIND(source='chunked'):
    paths = glob.glob(f'{source}/*.png')
    df = pd.DataFrame(data={'train': paths})
    # use re? Nah, this is chill, trust me bro
    def change(x):
        t = x.split('_')
        t[-2] = 'ISO200'
        return '_'.join(t)
    df['target'] = df['train'].apply(change)
    return df

class ReconDataset(torch.utils.data.Dataset):
    def __init__(self, df) -> None:
        super().__init__()
        self.ntf =  tf.Compose([
            tf.ToTensor(),
            tf.CenterCrop((256, 256))
            # tf.Normalize([79.85463183,  93.36660204, 110.08516866], [58.40277937, 59.24062922, 60.78025515])
        ])
        self.ctf = tf.Compose([
            tf.ToTensor(),
            tf.CenterCrop((256, 256))
            # tf.Normalize([81.10789587,  94.85670306, 111.62399867], [54.92694471, 57.5654794 , 58.92771677])
        ])
        self.df = df

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        # f = tf.Compose([
        #     tf.ToTensor(),
        #     # tf.Resize((256, 256))
        # ])
        x, y = self.df.iloc[idx]
        assert x.split('_')[-1] == y.split('_')[-1], 'Mismatch of chunks'
        assert y.split('_')[-2] == 'ISO200', 'Clean is incorrect'
        assert ''.join(x.split('_')[:-2]) == ''.join(y.split('_')[:-2]), 'Mismatch of item'
        nim = Image.open(x)
        cim = Image.open(y)
        # nim = cv2.imread(x)
        # cim = cv2.imread(y)
        # nim, cim = torch.FloatTensor(nim).permute(2, 0, 1), torch.FloatTensor(cim).permute(2, 0, 1)
        # print(nim.max(), nim.min(), cim.max(), cim.min())
        nim, cim = self.ntf(nim), self.ctf(cim)
        return nim, cim

    
class ReconDataModule(pl.LightningDataModule):
    def __init__(self, df, batch_size, check=False) -> None:
        super().__init__()
        self.train, self.test = train_test_split(df, test_size=0.2,
                                                 random_state=42)        
        self.batch_size = batch_size
        if check:
            self.checks()

    def checks(self):
        # mapped correctly?
        assert (self.train['train'].apply(lambda x: '_'.join(x.split('_')[:-2])) == 
                self.train['target'].apply(lambda x: '_'.join(x.split('_')[:-2]))).all(), 'Mismatches'
        assert (self.train['train'].apply(lambda x: '_'.join(x.split('_')[-1])) == 
                self.train['target'].apply(lambda x: '_'.join(x.split('_')[-1]))).all(), 'Mismatches'
        # all targets chunked correctly and exists?
        assert self.train['target'].map(lambda x: os.path.exists(x)).all()


    def train_dataloader(self):
        # assert all([j == i.replace('train', 'train_cleaned') for i, j in zip(n, c, strict=True)]), 'Mismatches!'
        return torch.utils.data.DataLoader(ReconDataset(self.train), 
                                           batch_size=self.batch_size,
                                           num_workers=8)
    
    def val_dataloader(self):
        # assert all([j == i.replace('train', 'train_cleaned') for i, j in zip(n, c, strict=True)]), 'Mismatches!'
        return torch.utils.data.DataLoader(ReconDataset(self.test), 
                                           batch_size=self.batch_size,
                                           num_workers=8)