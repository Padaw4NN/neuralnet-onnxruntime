import os
import torch
import mlflow
import mlflow.pytorch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataset import DataLoader
from dataset import get_classes
from dataset import create_dataloader
from mlp import CropRecommendationClassifier
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_step_MLP(
    model:        CropRecommendationClassifier,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    optimizer:    optim.Optimizer,
    criterion:    nn.CrossEntropyLoss,
    scheduler:    ReduceLROnPlateau,
    epochs:       int,
    device:       str
) -> None:
    """
    """
    torch.manual_seed(0)

    mlflow.set_tracking_uri(uri="http://0.0.0.0:8085")
    mlflow.set_experiment("crop-recommendation-classifier")

    with mlflow.start_run():

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", optimizer.param_groups[0]["lr"])
        mlflow.log_param("criterion", criterion.__class__.__name__)

        for epoch in range(epochs):
            running_loss_train   = 0.0
            running_loss_val     = 0.0
            running_accuracy_val = 0.0

            model.train()
            for X_train, y_train in train_loader:
                X_train, y_train = X_train.to(device), y_train.to(device)

                optimizer.zero_grad()

                output_train = model(X_train)
                loss_train = criterion(output_train, y_train)
                loss_train.backward()

                optimizer.step()

                running_loss_train += loss_train.item()
            
            avg_loss_train = running_loss_train / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss_train, step=epoch)

            total_samples_val = 0
            model.eval()
            with torch.inference_mode():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)

                    output_val = model(X_val)
                    loss_val = criterion(output_val, y_val)

                    running_loss_val += loss_val.item()
                    total_samples_val += y_val.size(0)

                    probs = F.softmax(output_val, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    running_accuracy_val += torch.sum((preds == y_val)).item()

            avg_loss_val = running_loss_val / len(val_loader)
            accuracy = running_accuracy_val / total_samples_val

            scheduler.step(avg_loss_val)

            print(
                f"[*] Epoch {epoch+1:3} of {epochs} | ",
                f"Loss: {running_loss_train:.5f} | ",
                f"Val Loss: {running_loss_val:.5f} | ",
                f"Val Accuracy: {accuracy * 100:.2f} % | ",
                f"LR: {optimizer.param_groups[0]['lr']:.6f} |",
            )

            mlflow.log_metric("val_loss", avg_loss_val, step=epoch)
            mlflow.log_metric("val_accuracy", accuracy, step=epoch)

    model_path = os.path.join("model", "model_last")
    os.makedirs(model_path, exist_ok=True)
    print(f"\n[!] Saving the torch model in the path: {model_path}/model_last.pth")
    torch.save(model.state_dict(), os.path.join(model_path, "model_last.pth"))
    
    print(f"[!] Exporting the model as ONNX Runtime in the path: {model_path}/model.onnx")
    torch.onnx.export(
        model.cpu(),
        (torch.rand((1, 7), dtype=torch.float32),),
        os.path.join(model_path, "model.onnx"),
        input_names=["input"]
    )

def run_train(epochs: int = 150) -> None:
    """
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = create_dataloader(
        fname="train.csv",
        batch_size=32,
        shuffle=True
    )

    val_loader = create_dataloader(
        fname="val.csv",
        batch_size=32,
        shuffle=True
    )

    input_size  = next(iter(train_loader))[0].shape[1]
    output_size = len(get_classes())

    model = CropRecommendationClassifier(
        input_size=input_size,
        num_classes=output_size
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        patience=10,
        factor=0.5
    )

    train_step_MLP(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=epochs,
        device=device
    )
