import os
import os.path as osp


class Config():
    def __init__(self) -> None:
        super().__init__()
        self.data_root = '/home/fyf/benke/GongShenghao/data/bladder_tumor_dataset/'
        # self.data_root = '/home/fyf/benke/Hec/data/bladder_tumor_data/'
        self.info_dir = osp.join(self.data_root, 'DataInfo.xlsx')
        self.val_ratio = 0.2
        self.results_root = '/home/fyf/benke/fyf/results'

    def chech_dir(self, d):
        if not osp.exists(d):
            print(f'making direction {d}!')
            os.makedirs(d)
        else:
            print(f'find direction {d}!')


class Config_Conb(Config):

    def __init__(self) -> None:
        super().__init__()
        self.grad_loss_ratio = 1.0
        self.stag_loss_ratio = 1.0
        self.results_root = osp.join(self.results_root, 'Conb')
        self.model_save_dir = osp.join(self.results_root, 'tumor_cls.pth')
        self.split_json = osp.join(self.results_root, 'split.json')
        self.chech_dir(self.results_root)


class Config_Split(Config):

    def __init__(self) -> None:
        super().__init__()
        self.results_root = osp.join(self.results_root, 'Split')
        self.model_save_dir = osp.join(self.results_root, 'tumor_cls.pth')
        self.split_json = osp.join(self.results_root, 'split.json')
        self.chech_dir(self.results_root)
