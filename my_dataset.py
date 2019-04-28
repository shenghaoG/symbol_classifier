import os.path as osp
from random import shuffle

import pandas as pd
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import Config

cfg = Config()

grad_map = {'Low': 0,
            'High': 1}
stag_map = {'MIBC': 0,
            'NMIBC': 1}

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def random_split(split_json, data_root, info_dir, val_ratio):
    if osp.exists(split_json):
        print(f'Load split file from {split_json}')
        with open(split_json) as f:
            lists = json.load(f)
        return lists['train_list'], lists['test_list']
    info = pd.read_excel(info_dir)
    img_list = list(info['ImageName'])
    grad_list = list(info['Grading'])
    stag_list = list(info['Staging'])
    img_list = [osp.join(data_root, 'TrainingData', x[1:-1] + '.jpg') for x in img_list]
    grad_list = [grad_map[x[1:-1]] for x in grad_list]
    stag_list = [stag_map[x[1:-1]] for x in stag_list]

    val_num = int(len(img_list) * val_ratio)
    idx_list = list(range(len(img_list)))
    shuffle(idx_list)

    train_idx = idx_list[val_num:]
    val_idx = idx_list[0:val_num]
    train_list = [{'img': img_list[x],
                   'grad': grad_list[x],
                   'stag': stag_list[x]} for x in train_idx]

    test_list = [{'img': img_list[x],
                 'grad': grad_list[x],
                 'stag': stag_list[x]} for x in val_idx]
    with open(split_json, 'w') as f:
        json.dump({'train_list': train_list,
                   'test_list': test_list}, f)
        print(f'save split file to {split_json}')
    return train_list, test_list


class Tumor_dataset(Dataset):

    def __init__(self, data_list, transform) -> None:
        super().__init__()
        self.transform = transform
        self.data_list = data_list

    def __getitem__(self, index):
        img = Image.open(self.data_list[index]['img']).convert('RGB')
        grad = self.data_list[index]['grad']
        stag = self.data_list[index]['stag']
        return self.transform(img), grad, stag

    def __len__(self):
        return len(self.data_list)


def load_dataset(split_json, data_root=cfg.data_root, info_dir=cfg.info_dir, val_ratio=cfg.val_ratio):
    train_list, val_list = random_split(split_json, data_root, info_dir, val_ratio)
    train_data = Tumor_dataset(train_list, data_transforms['train'])
    val_data = Tumor_dataset(val_list, data_transforms['val'])
    tumor_data = {'train': DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4),
                  'val': DataLoader(val_data, batch_size=4, shuffle=True, num_workers=4)}
    data_size = {'train': len(train_data),
                 'val': len(val_data)}
    return tumor_data, data_size


if __name__ == '__main__':
    load_dataset()
