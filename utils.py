import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import cv2


def plot_grad_flow(named_parameters):
    # plot that gradient flow
    f, ax = plt.subplots()
    f.tight_layout()
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.cpu().abs().mean())
            max_grads.append(p.grad.cpu().abs().max())

    print(ave_grads)

    ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    ax.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    # ax.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    ax.set_xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    ax.set_xlim(left=0, right=len(ave_grads))
    ax.set_ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    ax.set_xlabel("Layers")
    ax.set_ylabel("average gradient")
    ax.set_title("Gradient flow")
    ax.grid(True)
    ax.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    
    return f

def plot_samples(out, target, original):
    f, ax = plt.subplots(1, 3, figsize=(15,5))
    out = out[0].cpu().permute(1, 2, 0).detach().numpy()
    target = target[0].cpu().permute(1, 2, 0).numpy()
    original = original[0].cpu().permute(1, 2, 0).detach().numpy()
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    ax[0].imshow(out)
    ax[0].set_title('model')
    ax[1].imshow(np.clip(target, 0., 1.))
    ax[1].set_title('target')
    ax[2].imshow(np.clip(original-out, 0., 1.))
    ax[2].set_title('diff')

    return f