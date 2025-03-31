import os
import json
import torch
import argparse

from test import test
from predict import predict
from train_mlp import run_train
from dataset import get_classes


def main():
    parser = argparse.ArgumentParser(description="Crop Recommendation")
    parser.add_argument("--phase", type=str, help="Model phase [train|test|predict]")
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=150)
    args = parser.parse_args()

    if args.phase == "train":
        run_train(epochs=args.epochs)
    elif args.phase == "test":
        test()
    elif args.phase == "predict":
        with open(os.path.join("model", "data", "predict.json"), "r") as f:
            data = json.load(f)
        data = list(data.values())
        pred = predict(torch.tensor(data, dtype=torch.float32))
        print(f"[!] Predict: {pred} -> {get_classes()[pred]}")

if __name__ == "__main__":
    main()
