# %%
import matplotlib.pyplot as plt
import numpy as np

import torchvision

from parts import extract

# %%
data_dir = 'data/'
mnist_train = torchvision.datasets.MNIST(data_dir, train=True, download=False)

# %%
fours = [np.array(image) for image, label in mnist_train if label == 4]

# %%

fig, axes = plt.subplots(4, 4, sharex=True, sharey=True)
axes = axes.ravel()
for i, ax in enumerate(axes, start=16):
    image = fours[i]
    fe = extract.FourFeatures(fours[i])
    points = fe.bottom_pt, fe.top_left_pt, fe.top_right_pt, fe.middle_right_pt, fe.middle_left_pt
    x, y = zip(*points)

    ax.imshow(image, cmap='gray')
    ax.plot(y, x, '.r')
    ax.set_title(i)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

fig.tight_layout()
plt.show()

# %%
eights = [np.array(image) for image, label in mnist_train if label == 8]

# %%
fig, axes = plt.subplots(2, 2)
axes = axes.ravel()
for i, ax in enumerate(axes, start=32):
    image = eights[i]
    ax.imshow(image, cmap='gray')
    try:
        fe = extract.EightFeatures(image)
        fe = fe.bottom_pt, fe.top_pt
        x, y = zip(*fe)
        ax.plot(y, x, '.r')
    except ValueError:
        pass

    ax.set_title(i)
    ax.set_axis_off()

fig.tight_layout()
plt.show()

# %%
fives = [np.array(image) for image, label in mnist_train if label == 5]

# %%
fig, axes = plt.subplots(2, 2)
axes = axes.ravel()
for i, ax in enumerate(axes, start=20):
    image = fives[i]
    ax.imshow(image, cmap='gray')
    fe = extract.FiveFeatures(image)
    mask = np.zeros((*image.shape, 3), dtype=int)
    mask[fe.features['top']] = 0, 255, 0
    mask[fe.features['bottom']] += 0, 0, 255
    ax.imshow(mask, alpha=0.7)

    fe = fe.top_right, fe.top_left, fe.bottom_left, fe.end_circle
    x, y = zip(*fe)
    ax.plot(y, x, '.r')
    ax.set_title(i)
    ax.set_axis_off()

fig.tight_layout()
plt.show()

# %%
ones = [np.array(image) for image, label in mnist_train if label == 1]

# %%
fig, axes = plt.subplots(4, 4)
axes = axes.ravel()
for i, ax in enumerate(axes, start=32):
    image = ones[i]
    fe = extract.OneFeatures(image)
    points = [fe.top_pt, fe.bottom_pt]
    x, y = zip(*points)

    ax.imshow(image, cmap='gray')
    ax.plot(y, x, '.r')
    ax.set_axis_off()
    ax.set_title(i)

fig.tight_layout()
plt.show()

# %%
zeros = [np.array(image) for image, label in mnist_train if label == 0]

# %%
fig, axes = plt.subplots(4, 4)
axes = axes.ravel()
for i, ax in enumerate(axes, start=16):
    image = zeros[i]
    ax.imshow(image, cmap='gray')
    try:
        fe = extract.ZeroFeatures(image)
        ax.plot(fe.center_pt[1], fe.center_pt[0], '.r')
    except ValueError:
        pass

    ax.set_axis_off()
    ax.set_title(i)

fig.tight_layout()
plt.show()

# %%
