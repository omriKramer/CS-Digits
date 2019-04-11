#%%
import numpy as np
import torch
import torch.optim as optim
import torchvision

from digits.dataset import DigitsDataset
from digits import counter_stream
from digits.train import train_model
from digits import utils

#%%
data_dir = 'data/digits/'
transform = torchvision.transforms.Compose([
    torchvision.transforms.Pad((44, 2)),
    torchvision.transforms.ToTensor(),
])
train = DigitsDataset(data_dir + 'train', transform=transform)
test = DigitsDataset(data_dir + 'test', transform=transform)

loaders_dict = {
    'train': torch.utils.data.DataLoader(train, batch_size=4, shuffle=True),
    'val': torch.utils.data.DataLoader(test, batch_size=4, shuffle=True)
}

#%%
num_classes = 11
model = counter_stream.CounterStreamNet([2, 2, 2, 2], num_classes=num_classes)
optimizer = optim.Adam(model.parameters())

#%% Train the model
model, hist = train_model(model, loaders_dict, optimizer, num_classes, num_epochs=25)

#%% Alternatively load the model
checkpoint = torch.load('digits/checkpoint25.tar')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


#%%
data_iter = iter(loaders_dict['val'])
data = data_iter.next()

utils.imshow(torchvision.utils.make_grid(data['image'], nrow=1))

utils.imshow(np.vstack(data['segmentation'].numpy()), cmap='binary')


#%%
model.eval()
outputs = model(images).sigmoid()
for x in outputs:
    print(utils.one_hot_to_indices(x, confidence=0.8))


