import time
import copy

import torch
from torch import nn


def train_model(model, dataloaders, optimizer, num_classes, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    digits_loss = nn.BCEWithLogitsLoss()
    segmentation_loss = nn.MSELoss()

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

                y_onehot = torch.FloatTensor(len(image), num_classes).to(device)
                y_onehot.zero_()
                y_onehot.scatter_(1, digits, 1)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    bu_output = model(image, 'BU')
                    bu_loss = digits_loss(bu_output, y_onehot)

                    pred_segmentation = model(instruction, 'TD')
                    td_loss = segmentation_loss(pred_segmentation, segmentation)

                    loss = bu_loss + td_loss
                    pred_digits = bu_output.sigmoid() > 0.8

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * len(image)
                is_correct = torch.sum(pred_digits.eq(y_onehot.to(dtype=pred_digits.dtype)), 1) == 10
                running_corrects += torch.sum(is_correct)

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
