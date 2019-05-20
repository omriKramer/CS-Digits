# %%
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import corner_harris, corner_peaks

import skimage
from skimage.morphology import skeletonize
import torchvision

from points import extract
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
    interest_points = extract.FourInterestPoints(skeleton).points

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
        points = extract.EightInterestPoints(image)
        points = points.bottom, points.top
        x, y = zip(*points)
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
