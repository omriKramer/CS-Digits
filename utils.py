def np_image(x):
    return x.numpy().transpose(1, 2, 0).squeeze()


def plot_segmentation(ax, image, segmentation, ylabel=None):
    image = np_image(image)
    segmentation = np_image(segmentation)

    ax.imshow(image, cmap='gray')
    ax.imshow(segmentation, alpha=0.7, cmap='gray')
    ax.set_ylabel(ylabel)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
