import numpy as np

import torchvision
from torch.utils.data import Dataset

from parts import extract


class PointsDataset(Dataset):

    instructions = ['bottom', 'top_left', 'top_right', 'middle_left', 'middle_right']

    def __init__(self, root_dir, train=True, transform=None):
        """
        Args:
            root_dir: Directory with all the images.
            transform: Optional transform to be applied on the image.
        """
        self.transform = transform
        mnist_train = torchvision.datasets.MNIST(root_dir, train=train, download=False)
        self.fours = [np.array(image) for image, label in mnist_train if label == 4]
        self.points = [extract.FourInterestPoints(image > 120) for image in self.fours]

    def __len__(self):
        return len(self.points) * len(self.instructions)

    def __getitem__(self, idx):
        img_indx, instruction_idx = divmod(idx, len(self.instructions))

        image = self.fours[img_indx]
        points = self.points[img_indx]

        instruction = self.instructions[instruction_idx]
        point = points.__getattribute__(instruction)

        point_mask = np.zeros_like(image)
        i, j = point
        point_mask[i-1:i+2, j-1:j+2] = 255
        instruct_segment = (image > 0) * point_mask

        if self.transform:
            image = self.transform(image)
            instruct_segment = self.transform(instruct_segment)

        return {
            'image': image,
            'segmentation': instruct_segment,
            'instruction': instruction,
            'instruction_idx': instruction_idx,
            'point': np.array(point),
        }
