import os
import torch
import pandas as pd
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt

from mlp import CropRecommendationClassifier
from sklearn.metrics import confusion_matrix


def test() -> int:
    """
    """
    model_path = os.path.join("model", "model_last", "model_last.pth")
    
    data = pd.read_csv(
        os.path.join("model", "data", "test.csv"),
        engine='c',
        header=None
    )

    X = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(data.iloc[:, -1].values, dtype=torch.long)

    model = CropRecommendationClassifier(
        input_size=7,
        num_classes=22
    )
    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.inference_mode():
        output = model(X)
        probs = F.softmax(output, dim=1)
        preds = torch.argmax(probs, dim=1)
        corrects = torch.sum((preds == y)).item()
    accuracy = corrects / len(X)

    y = y.numpy()
    preds = preds.numpy()
    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, fmt='g', annot=True)
    plt.title(f"Test Crop Recommendation with accurracy: {accuracy * 100:.2f} %")
    plt.xlabel("y_pred")
    plt.ylabel("y_true")

    fig_path = os.path.join("model", "data", "imgs", "confusion_matrix.png")
    print(f"\n[!] Saving the figure in the path: {fig_path}")
    plt.savefig(fig_path)

    return preds, accuracy
