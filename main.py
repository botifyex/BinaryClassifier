import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

seed = 42
np.random.seed(seed)

points = np.random.rand(500, 2)
labels = (points[:, 0] + points[:, 1] > 1).astype(dtype=np.float32)

points_train, points_test, labels_train, labels_test = train_test_split(
    points, labels, test_size=0.2, random_state=seed
)

points_train = torch.FloatTensor(points_train)
points_test = torch.FloatTensor(points_test)
labels_train = torch.FloatTensor(labels_train).reshape(-1, 1)
labels_test = torch.FloatTensor(labels_test).reshape(-1, 1)

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()

        self.layer2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x

model = BinaryClassifier()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    outputs = model(points_train)

    loss: torch.Tensor = criterion(outputs, labels_train)
    loss.backward()

    optimizer.step()

    if epoch % 10 == 0:
        with torch.no_grad():
            test_outputs: torch.Tensor = model(points_test)
            test_loss: torch.Tensor = criterion(test_outputs, labels_test)
            
            predictions = (test_outputs > 0.5).float()
            accuracy = (predictions == labels_test).float().mean()

            print(
                f"Epoch: {epoch}, "
                f"Train Loss: {loss.item():.4f}, "
                f"Test Loss: {test_loss.item():.4f}, "
                f"Test Accuracy: {accuracy * 100:.2f}%"
            )

with torch.no_grad():
    test_point = torch.FloatTensor([[0.2412, 0.8124]])
    predictions: torch.Tensor = model(test_point)
    print(f"Predictions: {predictions.round().numpy()}")