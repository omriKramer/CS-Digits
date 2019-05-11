#%%
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%
num_classes = 11
model = counter_stream.CounterStreamNet([2, 2, 2, 2], num_classes=num_classes)
optimizer = optim.Adam(model.parameters())
model.to(device)

#%% Train the model
model, hist = train_model(model, loaders_dict, optimizer, num_classes, device, num_epochs=25)

#%% Alternatively load the model
checkpoint = torch.load('digits/model2.tar', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


#%%
data_iter = iter(loaders_dict['val'])
data = data_iter.next()

utils.imshow(torchvision.utils.make_grid(data['image'], nrow=1))

utils.imshow(torchvision.utils.make_grid(data['segmentation'], nrow=1))


#%%
model.clear()
model.eval()
bu1_output = model(data['image'], 'BU').sigmoid()
for x in bu1_output:
    print(utils.one_hot_to_indices(x, confidence=0.8))


#%%
with torch.no_grad():
    print(f'instructions are {data["instruction"]}')
    seg_out = model(data['instruction'], 'TD')
    utils.imshow(torchvision.utils.make_grid(seg_out, nrow=1))

#%%
bu2_output = model(data['image'], 'BU')
_, preds = torch.max(bu2_output, 1)
print(preds)
