import torch
import numpy as np
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, seed: int = 42):
        np.random.seed(seed)

        points = np.random.rand(500, 2)
        labels = (points[:, 0] + points[:, 1] > 1).astype(dtype=np.float32)

        points_train, points_test, labels_train, labels_test = train_test_split(
            points, labels, test_size=0.2, random_state=seed
        )

        self.points_train = torch.FloatTensor(points_train)
        self.points_test = torch.FloatTensor(points_test)
        self.labels_train = torch.FloatTensor(labels_train).reshape(-1, 1)
        self.labels_test = torch.FloatTensor(labels_test).reshape(-1, 1)