import matplotlib.pyplot as plt
import torch


def imshow(img, cmap=None):
    if isinstance(img, torch.Tensor):
        img = img.numpy().transpose(1, 2, 0)
    plt.imshow(img, cmap=cmap)
    plt.show()


def one_hot_to_indices(a, confidence=1):
    return (a >= confidence).nonzero().squeeze()
