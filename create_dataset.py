import random
from collections import defaultdict
from pathlib import Path
import string

import numpy as np
import PIL
import torchvision

data_dir = Path(__file__).parent / 'data'


def _generate_dataset(data, num_images, digits_per_image):
    groups = _group_by_digit(data)
    dataset = []
    for _ in range(num_images):
        labels = random.sample(groups.keys(), digits_per_image)
        images = [random.choice(groups[digit]) for digit in labels]
        new_image = np.hstack(images)
        new_image = PIL.Image.fromarray(new_image)
        dataset.append((new_image, labels))

    return dataset


def _group_by_digit(data):
    groups = defaultdict(list)
    for image, digit in data:
        groups[digit].append(image)

    return groups


def _generate_name(digits):
    filename = ''.join(map(str, digits)) + '_'
    filename += ''.join(random.sample(string.ascii_uppercase, 4))
    filename += '.png'
    return filename


def _write_dataset(data, directory, overwrite):
    try:
        directory.mkdir(parents=True)
    except FileExistsError:
        if not overwrite:
            raise
        for child in directory.iterdir():
            child.unlink()

    images, labels = zip(*data)
    image_names = [_generate_name(digits) for digits in labels]
    for image, name in zip(images, image_names):
        fp = directory / name
        image.save(fp)


def main(num_images, digits_per_image, overwrite=False):
    mnist_train = torchvision.datasets.MNIST(data_dir, train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(data_dir, train=False, download=True)

    train = _generate_dataset(mnist_train, num_images, digits_per_image)
    test = _generate_dataset(mnist_test, num_images // 6, digits_per_image)
    print('Writing training dataset...')
    dataset_dir = data_dir / 'digits'
    _write_dataset(train, dataset_dir / 'train', overwrite)

    print('writing test record...')
    _write_dataset(test, dataset_dir / 'test', overwrite)

    print('Done.')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create a dataset of randomly concatenated mnist images.')
    parser.add_argument('num_images', metavar='N', type=int, help='number of images in the created dataset')
    parser.add_argument('-d', '--digits', type=int, default=6, help='number of digits per image')
    parser.add_argument('-f', '--force', action='store_true', help='overwrite existing dataset')
    args = parser.parse_args()
    main(args.num_images, args.digits, args.force)
