#%%
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision

from digits.dataset import DigitsDataset
from digits import counter_stream
from digits.train import train_model
from digits import utils

#%%
data_dir = 'data/digits/'
train = DigitsDataset(data_dir + 'train', transform=torchvision.transforms.ToTensor())
test = DigitsDataset(data_dir + 'test', transform=torchvision.transforms.ToTensor())

loaders_dict = {
    'train': torch.utils.data.DataLoader(train, batch_size=4, shuffle=True),
    'val': torch.utils.data.DataLoader(test, batch_size=4, shuffle=True)
}

#%%
model = counter_stream.CounterStream(num_classes=10)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

#%% Train the model
model, hist = train_model(model, loaders_dict, criterion, optimizer, num_epochs=25)

#%% Alternatively load the model
checkpoint = torch.load('digits/checkpoint25.tar')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


#%%
data_iter = iter(loaders_dict['val'])
data = data_iter.next()

utils.imshow(torchvision.utils.make_grid(images, nrow=1))


#%%
model.eval()
outputs = model(images).sigmoid()
for x in outputs:
    print(utils.one_hot_to_indices(x, confidence=0.8))


