import cv2
import torch 
from torchvision import transforms as tf
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from main import ReconModel


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def infer(model_path, noisy, clean=None):
    # noisy is path, clean is path if for seeing corresponding clean im
    model = ReconModel.load_from_checkpoint(model_path)
    noisy_im = torch.FloatTensor(cv2.imread(noisy)/255.).permute(2, 0, 1).unsqueeze(0)
    clean_im = cv2.imread(clean)
    emb, clean_h = model(noisy_im)
    return clean_im, clean_h.squeeze(0).permute(1, 2, 0).detach().numpy()


if __name__ == '__main__':
    mpath = 'models/lightning_logs/version_14/checkpoints/epoch=14-step=885.ckpt'
    noisy = 'noisy/01_noisy_138.png'
    clean = 'clean/01_clean_138.png'
    clean_im, clean_h = infer(mpath, noisy, clean=clean)

    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(clean_im)
    ax[0].set_title('target')
    ax[1].imshow(clean_h)
    ax[1].set_title('model')
    plt.show()