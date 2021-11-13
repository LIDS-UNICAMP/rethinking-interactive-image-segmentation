import numpy as np
import torch as th
from torch.utils.data import Dataset, DataLoader
from loaders.maindata import MainData
from pytorch_metric_learning import samplers
from sklearn.preprocessing import LabelEncoder


class SimpleData(Dataset):
    def __init__(self, main_data: MainData):
        super().__init__()

        self.main_data = main_data
        self.components = []
        self.labels = []

        for comp in self.main_data:
            if comp.label != 'void':
                self.components.append(comp)
                self.labels.append(comp.label)

        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)
    
    def noise_sample(self, tensor, is_flip: bool = True):
        if is_flip:
            if np.random.choice(2):
                tensor = th.flip(tensor, (2,))  # CxHxW
        return tensor
        
    def __getitem__(self, index):
        return self.noise_sample(self.components[index].tensor), self.labels[index]

    def __len__(self):
        return len(self.components)


def get_balancedloader(main_data: MainData, class_size: int = 5, batch_size: int = 8) -> DataLoader:
    data = SimpleData(main_data)
    length = int(1e3)
    sampler = samplers.MPerClassSampler(data.labels, m=class_size, length_before_new_iter=length)
    return DataLoader(data, batch_size=batch_size * class_size, sampler=sampler)
