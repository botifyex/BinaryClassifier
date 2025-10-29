import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class PointsDataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.target[index]


class Data:
    def __init__(self, batch_size: int = 128, test_size: float = 0.2, seed: int = 42):
        np.random.seed(seed)

        self.batch_size = batch_size
        self.test_size = test_size
        self.seed = seed

        self.load()
        self.to_tensors()

        self.train_dataset = PointsDataset(self.features_train, self.target_train)
        self.test_dataset = PointsDataset(self.features_test, self.target_test)

        self.train_loader = DataLoader(self.train_dataset, self.batch_size)
        self.test_loader = DataLoader(self.test_dataset, self.batch_size)
    
    def load(self):
        features = np.random.rand(500, 2)
        target = (features[:, 0] + features[:, 1] > 1).astype(dtype=np.float32)

        self.features_train, self.features_test, self.target_train, self.target_test = train_test_split(
            features, target, test_size=self.test_size, random_state=self.seed
        )
    
    def to_tensors(self):
        self.features_train = torch.FloatTensor(self.features_train)
        self.features_test = torch.FloatTensor(self.features_test)
        self.target_train = torch.FloatTensor(self.target_train).unsqueeze(1)
        self.target_test = torch.FloatTensor(self.target_test).unsqueeze(1)