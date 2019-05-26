# %%
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import corner_harris, corner_peaks

from skimage.morphology import skeletonize
import torchvision

from parts import extract
# %%
data_dir = 'data/'
mnist_train = torchvision.datasets.MNIST(data_dir, train=True, download=False)

# %%
fours = [np.array(image) for image, label in mnist_train if label == 4]

# %%
titles = ['original', 'skeleton']

fig, axes = plt.subplots(4, len(titles), sharex=True, sharey=True)
for i, ax_row in enumerate(axes, start=24):
    image = fours[i]
    skeleton = skeletonize(image > 120)
    skeleton_corners = corner_peaks(corner_harris(skeleton), min_distance=2, exclude_border=False, num_peaks=5)
    interest_points = extract.FourFeatures(skeleton).points

    ax_row[0].imshow(image, cmap='gray')
    ax_row[0].set_ylabel(i)
    ax_row[0].plot(interest_points[:, 1], interest_points[:, 0], '.r')
    ax_row[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax_skeleton = ax_row[1]
    ax_skeleton.imshow(skeleton, cmap='gray')
    ax_skeleton.plot(interest_points[:, 1], interest_points[:, 0], '.r')
    ax_skeleton.axis('off')

for col, t in zip(axes[0], titles):
    col.set_title(t)

fig.tight_layout()
plt.show()

#%%
eights = [np.array(image) for image, label in mnist_train if label == 8]

#%%
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

#%%
fives = [np.array(image) for image, label in mnist_train if label == 5]

#%%
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

#%%
