import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import onnx
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def split_data(fname: str) -> None:
    """
    params:
    - fname: CSV file that will be split
    """
    base_path = os.path.join("model", "data")
    crop_path = os.path.join(base_path, fname)

    data = pd.read_csv(crop_path, engine='c')

    X = data.drop("Crop", axis=1).values
    y = data["Crop"].values

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Exporting the MinMaxScaler model to used it after on Java
    scaler_path = os.path.join("model", "model_last", "scaler.onnx")
    print(f"\n[*] Exporting the scaler as ONNX in the path: {scaler_path}")

    input_dim = 7
    onnx_scaler = convert_sklearn(scaler, initial_types=[("input", FloatTensorType([None, input_dim]))])
    onnx.save(onnx_scaler, scaler_path)

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=0.2,
        random_state=25
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=0.25,
        random_state=25
    )

    np.savetxt(os.path.join(base_path, "train.csv"), np.hstack((X_train, y_train.reshape(-1, 1))), delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(base_path, "val.csv"), np.hstack((X_val, y_val.reshape(-1, 1))), delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(base_path, "test.csv"), np.hstack((X_test, y_test.reshape(-1, 1))), delimiter=",", fmt="%.6f")

def get_classes() -> list:
    df = pd.read_csv("model/data/Crop_Recommendation.csv")
    le = LabelEncoder()
    df["le_Crop"] = le.fit_transform(df["Crop"])
    df = df.groupby(["Crop", "le_Crop"])["le_Crop"].value_counts().reset_index()
    keys = df["le_Crop"].values.tolist()
    values = df["Crop"].values
    ids_values = dict(zip(keys, values))
    return ids_values

def create_dataloader(fname: str, batch_size: int, shuffle: bool) -> DataLoader:
    """
    params:
    - fname: CSV file that will be loaded
    - batch_size: Number of samples per batch
    - shuffle: If data loader is shuffled

    returns: PyTorch dataloader
    """
    base_path = os.path.join("model", "data")
    data = pd.read_csv(os.path.join(base_path, fname), engine='c', header=None)

    X = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(data.iloc[:, -1].values, dtype=torch.long)

    dataloader = DataLoader(
        dataset=TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=4
    )

    return dataloader


if __name__ == "__main__":
    #split_data("Crop_Recommendation.csv")
    print(get_classes())