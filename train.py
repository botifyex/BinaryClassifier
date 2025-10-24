import torch
import torch.nn as nn

from src.dataset import Dataset
from src.model import BinaryClassifier

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
                    
                    predictions = test_outputs.round()
                    accuracy = (predictions == self.dataset.labels_test).float().mean()

                    print(
                        f"Epoch: {epoch}, "
                        f"Train Loss: {loss.item():.4f}, "
                        f"Test Loss: {test_loss.item():.4f}, "
                        f"Test Accuracy: {accuracy * 100:.2f}%"
                    )
    
    def predict(self, data: torch.FloatTensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(data)
    
    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()

    data = torch.FloatTensor([[0.1892414, 0.8114113]])
    predictions = trainer.predict(data)
    print(f"Predictions: {predictions.round().numpy()}")

    # trainer.save("assets/test-model.pth")