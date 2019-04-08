import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def one_hot_to_indices(a, confidence=1):
    return (a >= confidence).nonzero().squeeze()
