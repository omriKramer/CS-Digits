from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class DigitsDataset(Dataset):
    """Concatenated digits dataset."""

    def __init__(self, root_dir, num_digits=6, num_classes=11, digit_width=28, transform=None):
        """
        Args:
            root_dir: Directory with all the images.
            num_digits: Number of digits in an image.
            num_classes: Number of Classes including the "None" class.
            digit_width: Width of every digit inside the image.
            transofrm: Optional transform to be applied on the image. 
        """
        self.transform = transform
        self.digit_width = digit_width
        self.num_classes = num_classes
        self.num_digits = num_digits
        self.image_files = list(Path(root_dir).iterdir())

    def __len__(self):
        return len(self.image_files) * self.num_digits

    def __getitem__(self, idx):
        img_indx, instruction_idx = divmod(idx, self.num_digits)

        img_path = self.image_files[img_indx]
        image = Image.open(img_path)

        label_string, _, _ = img_path.name.partition('_')
        digits = np.array([int(digit) for digit in label_string])
        instruction = digits[instruction_idx]
        target_idx = instruction_idx + 1
        target = digits[target_idx] if target_idx < len(digits) else self.num_classes

        digit_mask = np.zeros_like(image)
        start_col = instruction_idx * self.digit_width
        digit_mask[:, start_col:start_col+self.digit_width] = 1
        instruct_segment = np.where(digit_mask, np.array(image) > 20, 0)

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'segmentation': instruct_segment,
            'digits': digits,
            'instruction': instruction,
            'target': target,
        }
