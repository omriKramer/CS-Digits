# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision
from torch import nn

import parts.dataset as ds
import utils
from parts.counter_stream import CounterStreamNet
from parts.train import train_model

# %%
data_dir = 'data/'
transform = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(lambda x: np.pad(x, 2, 'constant')),
    torchvision.transforms.ToTensor(),
])
train = ds.PartsDataset(data_dir, train=True, transform=transform)
test = ds.PartsDataset(data_dir, train=False, transform=transform)

loaders_dict = {
    'train': torch.utils.data.DataLoader(train, batch_size=4, shuffle=True),
    'val': torch.utils.data.DataLoader(test, batch_size=4, shuffle=True)
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
num_classes = 10
num_instructions = len(ds.feature2idx)
model = CounterStreamNet([2, 2, 2, 2], num_classes=num_classes, num_instructions=num_instructions)
model.to(device)
optimizer = optim.Adam(model.parameters())

# %% Train the model
bu_criterion = nn.CrossEntropyLoss()
td_criterion = nn.MSELoss()
model, hist = train_model(model, loaders_dict, optimizer, bu_criterion, td_criterion, device, num_epochs=2)

# %% Alternatively load the model
checkpoint = torch.load('parts/model.tar', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# %%
data_iter = iter(loaders_dict['val'])

# %%
for i, data in enumerate(data_iter):
    if i == 30:
        break
    model.clear()
    model.eval()
    with torch.no_grad():
        bu1_output = model(data['image'], 'BU').sigmoid()
        _, preds = bu1_output.max(1)

        td_out = model(data['instruction_idx'], 'TD')
        model_seg = td_out >= 0.6

    fig, axes = plt.subplots(len(model_seg), 2)
    for ax, img, gt, seg, instruction, pred in zip(axes, data['image'], data['segmentation'], model_seg,
                                                   data['instruction'], preds):

        utils.plot_segmentation(ax[0], img, gt, ylabel=instruction)
        utils.plot_segmentation(ax[1], img, seg, ylabel=pred.item())

    axes[0][0].set_title('Ground Truth')
    axes[0][1].set_title('Output')
    fig.tight_layout()
    # plt.show()
    fig.savefig(f'../../Desktop/results_110619/res{30+i}.png')

# %%
