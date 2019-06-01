import numpy as np

import torchvision
from torch.utils.data import Dataset

from parts import extract

digit2extractor = {
    0: extract.ZeroFeatures,
    1: extract.OneFeatures,
    4: extract.FourFeatures,
    5: extract.FiveFeatures,
    8: extract.EightFeatures,
}

features_table = {
    0: ('center',),
    1: ('top', 'bottom'),
    4: ('top_left', 'top_right', 'middle_left', 'middle_right', 'bottom'),
    5: ('top', 'bottom'),
    8: ('top', 'bottom'),
}


def _create_feature2idx():
    d = {}
    idx = 0
    for digit, features in features_table.items():
        for f in features:
            d[f'{digit}:{f}'] = idx
            idx += 1
    return d


feature2idx = _create_feature2idx()


class PartsDataset(Dataset):

    def __init__(self, root_dir, train=True, transform=None):
        """
        Args:
            root_dir: Directory with all the images.
            transform: Optional transform to be applied on the image.
        """
        self.transform = transform
        mnist = torchvision.datasets.MNIST(root_dir, train=train, download=False)
        self.items = []
        for image, label in mnist:
            image = np.array(image)
            try:
                fe = digit2extractor[label](image)
                image_features = [(image, label, feat, seg) for feat, seg in fe.features.items()]
                self.items.extend(image_features)
            except (KeyError, ValueError):
                pass

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image, label, instruction, segmentation = self.items[idx]
        segmentation = segmentation.astype(np.uint8) * 255
        instruction = f'{label}:{instruction}'
        instruction_idx = feature2idx[instruction]

        if self.transform:
            image = self.transform(image)
            segmentation = self.transform(segmentation)

        return {
            'image': image,
            'label': label,
            'segmentation': segmentation,
            'instruction': instruction,
            'instruction_idx': instruction_idx,
            'idx': idx,
        }
