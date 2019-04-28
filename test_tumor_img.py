from __future__ import print_function, division

import time

import torch

from config import Config_Conb
from my_dataset import load_dataset
from my_model import resnet18, tt

cfg = Config_Conb()
dataloaders, dataset_sizes = load_dataset(cfg.split_json)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_model(model):
    since = time.time()

    print('Test Start!')
    print('-' * 10)

    model.eval()  # Set model to evaluate mode

    corrects_grad = 0
    corrects_stag = 0

    # Iterate over data.
    for inputs, labels_grad, labels_stag in dataloaders['val']:
        inputs = inputs.to(device)
        labels_grad = labels_grad.to(device)
        labels_stag = labels_stag.to(device)

        # forward
        # track history if only in train
        outputs_grad, outputs_stag = model(inputs)
        _, preds_grad = torch.max(outputs_grad, 1)
        _, preds_stag = torch.max(outputs_stag, 1)

        # statistics
        corrects_grad += torch.sum(preds_grad == labels_grad.data)
        corrects_stag += torch.sum(preds_stag == labels_stag.data)

    acc_grad = corrects_grad.double() / dataset_sizes['val']
    acc_stag = corrects_stag.double() / dataset_sizes['val']

    time_elapsed = time.time() - since
    print(f'Testing complete in {time_elapsed // 60:.0f}m '
          f'{time_elapsed % 60:.0f}s')
    print(f'Test grad acc: {acc_grad:4f} stag acc: {acc_stag:.4f}')


if __name__ == '__main__':
    # Load a pretrained model and reset final fully connected layer.
    model_resnet18 = resnet18(pretrained=False)
    model_hp = tt(model_resnet18, 2, 2)
    # Load Trained Model
    model_hp.load_state_dict(torch.load(cfg.model_save_dir))
    print(f'Loading Model from {cfg.model_save_dir}!')
    model_hp = model_hp.to(device)

    test_model(model_hp)
    print(f'Done!')
