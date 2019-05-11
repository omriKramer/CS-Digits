#%%
import numpy as np

import torch
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
optimizer = optim.Adam(model.parameters())
model.to(device)

#%% Train the model
model, hist = train_model(model, loaders_dict, optimizer, device, num_epochs=2)
