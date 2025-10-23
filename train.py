import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

from src.dataset import Dataset
from src.model import BinaryClassifier

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

class Trainer:
    def __init__(self):
        self.model = BinaryClassifier()
        self.dataset = Dataset()
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
    
    def run(self, epochs: int = 100):
        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()
            outputs = self.model(self.dataset.points_train)

            loss: torch.Tensor = self.criterion(outputs, self.dataset.labels_train)
            loss.backward()

            self.optimizer.step()

            if epoch % 10 == 0:
                with torch.no_grad():
                    test_outputs: torch.Tensor = self.model(self.dataset.points_test)
                    test_loss: torch.Tensor = self.criterion(test_outputs, self.dataset.labels_test)
                    
                    predictions = (test_outputs > 0.5).float()
                    accuracy = (predictions == self.dataset.labels_test).float().mean()

                    print(
                        f"Epoch: {epoch}, "
                        f"Train Loss: {loss.item():.4f}, "
                        f"Test Loss: {test_loss.item():.4f}, "
                        f"Test Accuracy: {accuracy * 100:.2f}%"
                    )
    
    def predict(self, data: torch.FloatTensor):
        with torch.no_grad():
            predictions: torch.Tensor = self.model(data)
            print(f"Predictions: {predictions.round().numpy()}")
    
    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()

    trainer.predict(torch.FloatTensor([
        [0.1892414, 0.8114113]
    ]))

    trainer.save("assets/model-test.pth")