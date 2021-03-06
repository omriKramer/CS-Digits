import time
import copy

import torch

SMOOTH = 1e-6


def train_model(model, dataloaders, optimizer, bu_criterion, td_criterion, device, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_iou = 0

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
            running_iou = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                image = data['image'].to(device)
                segmentation = data['segmentation'].to(device)
                instruction_idx = data['instruction_idx'].to(device)
                labels = data['label'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # clear model inner state
                model.clear()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    pred_output = model(image, 'BU')
                    pred_loss = bu_criterion(pred_output, labels)

                    pred_segmentation = model(instruction_idx, 'TD')
                    seg_loss = td_criterion(pred_segmentation, segmentation)

                    loss = pred_loss + seg_loss

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * len(image)

                _, preds = torch.max(pred_output, 1)
                running_corrects += torch.sum(preds == labels.data)

                pred_segmentation = pred_segmentation > 0.5
                segmentation = segmentation.to(torch.uint8)
                union = (pred_segmentation | segmentation).float().sum((1, 2, 3))
                intersection = (pred_segmentation & segmentation).float().sum((1, 2, 3))
                iou = (intersection + SMOOTH) / (union + SMOOTH)
                running_iou += iou.sum()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_iou = running_iou / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} IoU: {epoch_iou:.4f}, Class Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_iou > best_iou:
                best_iou = epoch_iou
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append((epoch_iou, epoch_acc))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f'Best val IoU: {best_iou:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history
