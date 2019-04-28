from __future__ import print_function, division

import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from my_model import resnet18new
from my_dataset import load_dataset
from config import Config_Split

cfg = Config_Split()
dataloaders, dataset_sizes = load_dataset(cfg.split_json)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = {x: copy.deepcopy(model[x].state_dict())
                      for x in ['grading', 'staging']}
    best_acc = {'grading': 0.0,
                'staging': 0.0}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':

                for x in ['grading', 'staging']:
                    scheduler[x].step()
                    model[x].train()  # Set model to training mode
            else:
                for x in ['grading', 'staging']:
                    model[x].eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = {'grading': 0,
                                'staging': 0}

            # Iterate over data.
            for inputs, labels_grad, labels_stag in dataloaders[phase]:
                inputs = inputs.to(device)
                labels_grad = labels_grad.to(device)
                labels_stag = labels_stag.to(device)
                labels = {'grading': labels_grad,
                          'staging': labels_stag}

                # zero the parameter gradients
                for x in ['grading', 'staging']:
                    optimizer[x].zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = {x: model[x](inputs)
                               for x in ['grading', 'staging']}
                    _, preds_grad = torch.max(outputs['grading'], 1)
                    _, preds_stag = torch.max(outputs['staging'], 1)
                    preds = {'grading': preds_grad,
                             'staging': preds_stag}
                    loss = {x: criterion(outputs[x], labels[x])
                            for x in ['grading', 'staging']}
                    loss = loss['grading'] + loss['staging']

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        for x in ['grading', 'staging']:
                            optimizer[x].step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                corrects = {x: torch.sum(preds[x] == labels[x])
                            for x in ['grading', 'staging']}
                running_corrects['grading'] += corrects['grading']
                running_corrects['staging'] += corrects['staging']
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = {x: running_corrects[x].double() / dataset_sizes[phase]
                         for x in ['grading', 'staging']}
            for x in ['grading', 'staging']:
                print(x + '{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc[x]))

            # deep copy the model
            for x in ['grading', 'staging']:
                if phase == 'val' and epoch_acc[x] > best_acc[x]:
                    best_acc[x] = epoch_acc[x]
                    best_model_wts[x] = copy.deepcopy(model[x].state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    for x in ['grading', 'staging']:
        print(x + 'Best val Acc: {:4f}'.format(best_acc[x]))

    # load best model weights
    model_ft_grad.load_state_dict(best_model_wts['grading'])
    model_ft_stag.load_state_dict(best_model_wts['staging'])
    return model_ft_grad, model_ft_stag


# Load a pretrained model and reset final fully connected layer.

model_ft_grad, model_ft_stag = resnet18new(pretrained=True)
model_ft = {'grading': model_ft_grad,
            'staging': model_ft_stag}
for x in ['grading', 'staging']:
    num_ftrs = model_ft[x].fc.in_features
    model_ft[x].fc = nn.Linear(num_ftrs, 2)
    model_ft[x] = model_ft[x].to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = {x: optim.SGD(model_ft[x].parameters(), lr=0.001, momentum=0.9)
                for x in ['grading', 'staging']}

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = {x: lr_scheduler.StepLR(optimizer_ft[x], step_size=7, gamma=0.1)
                    for x in ['grading', 'staging']}

model_ft_grad, model_ft_stag = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                           num_epochs=25)
