import torch
import torch.nn as nn

from src.data import Data
from src.model import BinaryClassifier

class Trainer(Data):
    def __init__(self, batch_size = 128, test_size = 0.2, seed = 42):
        super().__init__(batch_size, test_size, seed)

        self.model = BinaryClassifier()

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01, weight_decay=1e-3)
    
    def train(self):
        train_loss = 0.0
        train_batches = 0

        self.model.train()

        for features_batch, target_batch in self.train_loader:
            outputs = self.model(features_batch)

            loss: torch.Tensor = self.criterion(outputs, target_batch)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += loss.item()
            train_batches += 1
        
        return train_loss, train_batches
    
    def eval(self):
        test_loss = 0.0
        test_batches = 0
        test_mae = 0.0

        self.model.eval()

        with torch.no_grad():
            for features_batch, target_batch in self.test_loader:
                outputs: torch.Tensor = self.model(features_batch)
                loss: torch.Tensor = self.criterion(outputs, target_batch)

                test_mae += torch.abs(outputs - target_batch).mean().item()

                test_loss += loss.item()
                test_batches += 1
        
        return test_loss, test_batches, test_mae

    def run(self, epochs: int = 100):
        for epoch in range(1, epochs + 1):
            train_loss, train_batches = self.train()

            if epoch % 10 == 0:
                test_loss, test_batches, test_mae = self.eval()

                average_train_loss = train_loss / train_batches
                average_test_loss = test_loss / test_batches
                average_test_mae = test_mae / test_batches

                print(
                    f"Epoch: {epoch}, "
                    f"Train Loss: {average_train_loss:.4f}, "
                    f"Test Loss: {average_test_loss:.4f}, "
                    f"Test MAE: {average_test_mae:.3f}"
                )
    
    def predict(self, data: torch.FloatTensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(data)
    
    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()