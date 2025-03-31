import torch
import torch.nn as nn


class CropRecommendationClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int) -> None:
        super(CropRecommendationClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)

        self.fc5 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.1)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.relu(self.bn1(self.fc1(X)))
        X = self.dropout(X)
        X = self.relu(self.bn2(self.fc2(X)))
        X = self.dropout(X)
        X = self.relu(self.bn3(self.fc3(X)))
        X = self.dropout(X)
        X = self.relu(self.bn4(self.fc4(X)))
        X = self.fc5(X)
        return X
