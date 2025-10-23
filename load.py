import torch
from src.model import BinaryClassifier

def main():
    model = BinaryClassifier()
    model.load_state_dict(torch.load("assets/model.pth"))
    model.eval()

    with torch.no_grad():
        data = torch.FloatTensor([[0.1892414, 0.8114113]])
        predictions: torch.Tensor = model(data)
        print(f"Predictions: {predictions.round().numpy()}")

if __name__ == '__main__':
    main()