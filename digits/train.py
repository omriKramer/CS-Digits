import time
import copy

import torch
from torch import nn


def train_model(model, dataloaders, optimizer, num_classes, device, num_epochs=25):
    digits_criterion = nn.BCEWithLogitsLoss()
    segmentation_criterion = nn.MSELoss()
    target_critertion = nn.CrossEntropyLoss()

    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                image = data['image'].to(device)
                digits = data['digits'].to(device)
                segmentation = data['segmentation'].to(device)
                instruction = data['instruction'].to(device)
                target = data['target'].to(device)

                y_onehot = torch.FloatTensor(len(image), num_classes).to(device)
                y_onehot.zero_()
                y_onehot.scatter_(1, digits, 1)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    model.clear()

                    bu_output = model(image, 'BU')
                    digits_loss = digits_criterion(bu_output, y_onehot)

                    pred_segmentation = model(instruction, 'TD')
                    segmentation_loss = segmentation_criterion(pred_segmentation, segmentation)

                    bu2_output = model(image, 'BU')
                    target_loss = target_critertion(bu2_output, target)

                    loss = digits_loss + segmentation_loss + target_loss
                    _, preds = torch.max(bu2_output, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * len(image)
                running_corrects += torch.sum(preds == target)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history
