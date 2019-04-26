from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import xlrd
import pandas
import random
import os
import os.path as osp
import pandas as pd
from random import shuffle

#data_root = '/home/fengyifan/data/tiaotiao/tumor_cls/tumor_data'
#data_root = '/home/fyf/benke/Hec/data/'#数据根地址
data_root = '/home/fyf/benke/GongShenghao/data/trash_data'
info_dir = osp.join(data_root, 'DataInfo.xlsx')
val_ratio = 0.2

type_map = {'cardboard': 0,
            'paper': 1,
            'glass':2,
            'metal':3,
            'plastic':4}

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

def random_split(data_root=data_root,#数据随机化
                 info_dir=info_dir,
                 val_ratio=val_ratio):
    info = pd.read_excel(info_dir)
    img_list = list(info['ImageName'])
    type_list = list(info['Tpye'])
    img_list = [osp.join(data_root, 'TrainingData', 'trash(' + str(x) + ').jpg') for x in img_list]
    type_list = [type_map[x] for x in type_list]

    val_num = int(len(img_list) * val_ratio)
    idx_list = list(range(len(img_list)))
    shuffle(idx_list)#随机化idx_list

    train_idx = idx_list[val_num:]#train取后0.8的部分
    val_idx = idx_list[0:val_num]#val取前0。2的部分
    train_list = [{'img': img_list[x],
                   'type': type_list[x]} for x in train_idx]

    val_list = [{'img': img_list[x],
                 'type': type_list[x]} for x in val_idx]
    return train_list, val_list

class Trash_dataset(Dataset):

    def __init__(self, data_list, transform) -> None:
        super().__init__()
        self.transform = transform
        self.data_list = data_list

    def __getitem__(self, index):
        img = Image.open(self.data_list[index]['img']).convert('RGB')
        type = self.data_list[index]['type']
        return self.transform(img), type

    def __len__(self):
        return len(self.data_list)


def load_dataset(data_root=data_root, info_dir=info_dir, val_ratio=val_ratio):
    train_list, val_list = random_split(data_root, info_dir, val_ratio)
    train_data = Trash_dataset(train_list, data_transforms['train'])
    val_data = Trash_dataset(val_list, data_transforms['val'])
    tumor_data = {'train': DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4),
                  'val': DataLoader(val_data, batch_size=16, shuffle=True, num_workers=4)}
    data_size = {'train': len(train_data),
                 'val': len(val_data)}
    return tumor_data, data_size

