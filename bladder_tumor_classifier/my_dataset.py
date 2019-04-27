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

data_root = '/home/fyf/benke/GongShenghao/data/bladder_tumor_dataset/'
# data_root = '/home/fyf/benke/Hec/data/bladder_tumor_data/'
info_dir = osp.join(data_root, 'DataInfo.xlsx')
val_ratio = 0.2

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


def random_split(data_root=data_root,
                 info_dir=info_dir,
                 val_ratio=val_ratio):
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

    val_list = [{'img': img_list[x],
                 'grad': grad_list[x],
                 'stag': stag_list[x]} for x in val_idx]
    return train_list, val_list


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


def load_dataset(data_root=data_root, info_dir=info_dir, val_ratio=val_ratio):
    train_list, val_list = random_split(data_root, info_dir, val_ratio)
    train_data = Tumor_dataset(train_list, data_transforms['train'])
    val_data = Tumor_dataset(val_list, data_transforms['val'])
    tumor_data = {'train': DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4),
                  'val': DataLoader(val_data, batch_size=4, shuffle=True, num_workers=4)}
    data_size = {'train': len(train_data),
                 'val': len(val_data)}
    return tumor_data, data_size


# def write_txt(txt_path=None, cls=None, data_path=None, num=0):
#     rand_separate = np.array(random.sample(range(0, num), num))
#     mid = int(num / 2)
#     f = open(txt_path + 'train.txt', 'w')
#     for i in rand_separate[np.arange(0, mid)]:
#         img_path = data_path + 'image (' + str(i + 1) + ').jpg'
#         f.write(img_path + ' ' + str(cls[i]) + '\n')
#     f.close()
#     f = open(txt_path + 'test.txt', 'w')
#     for i in rand_separate[np.arange(mid, num)]:
#         img_path = data_path + 'image (' + str(i + 1) + ').jpg'
#         f.write(img_path + ' ' + str(cls[i]) + '\n')
#     f.close()
#
#
# def pre_load(data_path=None, label_list_name=None):
#     _num_images_train = 478
#     os.mkdir(data_path + './catalogue')
#     os.mkdir(data_path + 'catalogue/' + './grading')
#     os.mkdir(data_path + 'catalogue/' + './staging')
#     excel_file = xlrd.open_workbook(data_path + label_list_name)
#     sheet = excel_file.sheet_by_index(0)
#     label_grading = sheet.col_values(1)
#     label_staging = sheet.col_values(2)
#     cls_grading = pandas.Categorical(label_grading)[np.arange(1, _num_images_train + 1)].codes
#     cls_staging = pandas.Categorical(label_staging)[np.arange(1, _num_images_train + 1)].codes
#     write_txt(txt_path=data_path + 'catalogue/grading/', cls=cls_grading, data_path=data_path, num=_num_images_train)
#     write_txt(txt_path=data_path + 'catalogue/staging/', cls=cls_staging, data_path=data_path, num=_num_images_train)
#
#
# def default_loader(path):
#     return Image.open(path).convert('RGB')
#
#
# class MyDataset(Dataset):
#     def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
#         fh = open(txt, 'r')
#         imgs = []
#         for line in fh:
#             line = line.strip('\n')
#             line = line.rstrip()
#             words = line.split()
#             imgs.append((words[0] + ' ' + words[1], int(words[2])))
#         self.imgs = imgs
#         self.transform = transform
#         self.target_transform = target_transform
#         self.loader = loader
#
#     def __getitem__(self, index):
#         fn, label = self.imgs[index]
#         img = self.loader(fn)
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, label
#
#     def __len__(self):
#         return len(self.imgs)
#
#
# def _load_dataset(data_path=None, label_list_name=None):
#     data_path = data_path
#     label_list_name = label_list_name
#     pre_load(data_path=data_path, label_list_name=label_list_name)
#     data_transforms = {
#         'train': transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'val': transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#     }
#     grading_train_data = MyDataset(txt=data_path + 'catalogue/grading/train.txt', transform=data_transforms['train'])
#     grading_test_data = MyDataset(txt=data_path + 'catalogue/grading/test.txt', transform=data_transforms['val'])
#     staging_train_data = MyDataset(txt=data_path + 'catalogue/staging/train.txt', transform=data_transforms['train'])
#     staging_test_data = MyDataset(txt=data_path + 'catalogue/staging/test.txt', transform=data_transforms['val'])
#     grading_data = {'train': DataLoader(grading_train_data, batch_size=4, shuffle=True, num_workers=4),
#                     'val': DataLoader(grading_test_data, batch_size=4, shuffle=True, num_workers=4)}
#     staging_data = {'train': DataLoader(staging_train_data, batch_size=4, shuffle=True, num_workers=4),
#                     'val': DataLoader(staging_test_data, batch_size=4, shuffle=True, num_workers=4)}
#     grading_data_sizes = {'train': len(grading_train_data), 'val': len(grading_test_data)}
#     staging_data_sizes = {'train': len(staging_train_data), 'val': len(staging_test_data)}
#     return grading_data, staging_data, grading_data_sizes, staging_data_sizes
#

if __name__ == '__main__':
    load_dataset()
