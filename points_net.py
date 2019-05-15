#%%
import itertools

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import torch
from torch import nn
import torch.optim as optim
import torchvision

from points.counter_stream import CounterStreamNet
from points.dataset import PointsDataset
from points.train import train_model

#%%
data_dir = 'data/'
transform = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(lambda img: np.pad(img, 2, 'constant')),
    torchvision.transforms.ToTensor(),
])
train = PointsDataset(data_dir, train=True,  transform=transform)
test = PointsDataset(data_dir, train=False, transform=transform)

loaders_dict = {
    'train': torch.utils.data.DataLoader(train, batch_size=4, shuffle=True),
    'val': torch.utils.data.DataLoader(test, batch_size=4, shuffle=True)
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%
num_classes = 10
num_instructions = len(train.instructions)
model = CounterStreamNet([2, 2, 2, 2], num_classes=num_classes, num_instructions=num_instructions)
model.to(device)
optimizer = optim.Adam(model.parameters())

#%% Train the model
criterion = nn.MSELoss()
model, hist = train_model(model, loaders_dict, optimizer, criterion, device, num_epochs=2)

#%% Alternatively load the model
checkpoint = torch.load('points/model_mse.tar', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#%%
data_iter = iter(loaders_dict['val'])

#%%
for i, data in enumerate(data_iter):
    if i == 10:
        break
    model.clear()
    model.eval()
    with torch.no_grad():
        bu1_output = model(data['image'], 'BU').sigmoid()
        seg_out = model(data['instruction_idx'], 'TD')

    fig, axes = plt.subplots(len(seg_out), 2)
    for ax, img, gt, seg, instruction in zip(axes, data['image'], data['segmentation'], seg_out, data['instruction']):
        img = img.numpy().transpose(1, 2, 0).squeeze()
        gt = gt.numpy().transpose(1, 2, 0).squeeze()
        seg = seg.numpy().transpose(1, 2, 0).squeeze()
        seg = np.where(seg < 0.6, 0, 1)

        ax[0].imshow(img, cmap='gray')
        ax[0].imshow(gt, alpha=0.7, cmap='gray')
        ax[0].set_ylabel(instruction)

        ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        ax[1].imshow(img, cmap='gray')
        ax[1].imshow(seg, alpha=0.7, cmap='gray')
        ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    axes[0][0].set_title('Ground Truth')
    axes[0][1].set_title('Output')
    fig.tight_layout()
    # plt.show()
    fig.savefig(f'../../Desktop/res{i}.png')
#%%
model.eval()
results = []
for data in loaders_dict['val']:
    with torch.no_grad():
        model.clear()
        model(data['image'], 'BU')
        seg_out = model(data['instruction_idx'], 'TD').squeeze()
        seg_out = seg_out > 0.5
        gt = data['segmentation'].to(torch.uint8).squeeze()
        results.append((seg_out, gt, data['instruction']))

#%%
model.eval()
iou = []
for seg_out, gt, _ in results:
    union = seg_out | gt
    intersection = seg_out & gt
    current_iou = intersection.sum((1, 2), dtype=torch.float) / union.sum((1, 2), dtype=torch.float)
    iou.append(current_iou)

iou = torch.cat(iou)
print(iou.mean())

#%%
_, _, inst = zip(*results)
inst = list(itertools.chain.from_iterable(inst))

#%%
s = pd.DataFrame({'iou': iou, 'instruction': inst})
print(s.groupby('instruction').mean())
