import torch

CURRENT = 'davis'
DIRS = {
    'davis': ['<your path>/DAVIS/img',
              '<your path>/DAVIS/boundaries',
              '<your path>/DAVIS/gt'],
}


class Config:
    @staticmethod
    def current_dataset():
        return CURRENT

    @staticmethod
    def images_dir():
        return DIRS[CURRENT][0]

    @staticmethod
    def boundaries_dir():
        return DIRS[CURRENT][1]

    @staticmethod
    def groundtruth_dir():
        return DIRS[CURRENT][2]

    @staticmethod
    def is_cityscapes():
        return CURRENT == 'cityscapes'

    @staticmethod
    def pickle_path():
        return 'app.pickle'

    @staticmethod
    def cuda():
        return torch.cuda.is_available()

    @staticmethod
    def sample():
        """Between 0 and 1 or None"""
        return None

    @staticmethod
    def n_jobs():
        """Integer or None"""
        return 7

    @staticmethod
    def preload():
        return False

    @staticmethod
    def batch_load_size():
        return 15

    @staticmethod
    def pattern():
        return r'.'

    @staticmethod
    def batch_size():
        return 64

    @staticmethod
    def embedding_network():
        return 'hrnet'

    @staticmethod
    def use_gt():
        return False

    @staticmethod
    def save_masks_path():
        return 'pred'

    @staticmethod
    def weights_path():
        return 'weights/hrnet_w18_small_model_v1.pth'

    @staticmethod
    def hrnet_config_path():
        return 'src/networks/config/seg_hrnet_w18_small_v1_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml'
