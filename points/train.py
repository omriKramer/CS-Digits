import time
import copy
import math

import torch
from torch import nn


def train_model(model, dataloaders, optimizer, device, num_epochs=25):
    segmentation_criterion = nn.MSELoss()

    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = math.inf

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for data in dataloaders[phase]:
                image = data['image'].to(device)
                segmentation = data['segmentation'].to(device)
                instruction_idx = data['instruction_idx'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # clear model inner state
                model.clear()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    model(image, 'BU')

                    pred_segmentation = model(instruction_idx, 'TD')
                    loss = segmentation_criterion(pred_segmentation, segmentation)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * len(image)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f'Best val loss: {best_loss:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history
