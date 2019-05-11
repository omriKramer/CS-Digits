# %%
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import corner_harris, corner_peaks

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
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line

image = skeleton = skeletonize(fours[0] > 120)
h, theta, d = hough_line(image)

# Generating figure 1
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Input image')
ax[0].set_axis_off()

ax[1].imshow(np.log(1 + h),
             extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
             cmap='gray', aspect=1/1.5)
ax[1].set_title('Hough transform')
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')
ax[1].axis('image')

ax[2].imshow(image, cmap='gray')
for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
    ax[2].plot((0, image.shape[1]), (y0, y1), '-r')
ax[2].set_xlim((0, image.shape[1]))
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_axis_off()
ax[2].set_title('Detected lines')

plt.tight_layout()
plt.show()

#%%
image = fours[0]
lines = probabilistic_hough_line(image, threshold=10, line_length=4,  line_gap=5)

# Generating figure 2
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Input image')

ax[1].imshow(image * 0)
for line in lines:
    p0, p1 = line
    ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[1].set_xlim((0, image.shape[1]))
ax[1].set_ylim((image.shape[0], 0))
ax[1].set_title('Probabilistic Hough')

for a in ax:
    a.set_axis_off()

plt.tight_layout()
plt.show()
