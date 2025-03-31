import os
import torch
import torch.nn.functional as F

from mlp import CropRecommendationClassifier


def predict(X: torch.Tensor) -> int:
    """
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join("model", "model_last", "model_last.pth")
    
    model = CropRecommendationClassifier(
        input_size=7,
        num_classes=22
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    model.eval()
    with torch.inference_mode():
        output = model(X.unsqueeze(0).to(device))
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).detach().cpu().item()
    
    return pred
