from __future__ import print_function, division

import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from bladder_tumor_classifier import load_dataset
from bladder_tumor_classifier import resnet18, tt

grad_loss_ratio = 1.0
stag_loss_ratio = 1.0
# plt.ion()  # interactive mode
# data_path = '/home/fyf/benke/Hec/data/bladder_tumor_data/'
# label_list_name = 'DataInfo.xlsx'
dataloaders, dataset_sizes = load_dataset()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, num_epochs=40):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc_grad = 0.0
    best_acc_stag = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects_grad = 0
            running_corrects_stag = 0

            # Iterate over data.
            for inputs, labels_grad, labels_stag in dataloaders[phase]:
                inputs = inputs.to(device)
                labels_grad = labels_grad.to(device)
                labels_stag = labels_stag.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs_grad, outputs_stag = model(inputs)
                    _, preds_grad = torch.max(outputs_grad, 1)
                    _, preds_stag = torch.max(outputs_stag, 1)
                    loss_grad = criterion(outputs_grad, labels_grad)
                    loss_stag = criterion(outputs_stag, labels_stag)
                    loss = grad_loss_ratio * loss_grad + stag_loss_ratio * loss_stag

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects_grad += torch.sum(preds_grad == labels_grad.data)
                running_corrects_stag += torch.sum(preds_stag == labels_stag.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc_grad = running_corrects_grad.double() / dataset_sizes[phase]
            epoch_acc_stag = running_corrects_stag.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} '
                  f'Acc grad: {epoch_acc_grad:.4f} Acc stag {epoch_acc_stag:.4f}')

            # deep copy the model
            if phase == 'val' and \
                    epoch_acc_grad > best_acc_grad and epoch_acc_stag > best_acc_stag:
                best_acc_grad = epoch_acc_grad
                best_acc_stag = epoch_acc_stag
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m '
          f'{time_elapsed % 60:.0f}s')
    print(f'Best val grad acc: {best_acc_grad:4f} stag acc: {best_acc_stag:.4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Load a pretrained model and reset final fully connected layer.

model_resnet18 = resnet18(pretrained=True)
model_hp = tt(model_resnet18, 2, 2)
model_hp = model_hp.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_hp.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_hp, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)
