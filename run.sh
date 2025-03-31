#!/bin/bash
phase=$1
epochs=$2

if [[ "$phase" == "train" ]]; then
    if [[ -z "$epochs" ]]; then
        epochs=150
    fi

    echo -e "\n[*] Executing the training step with $epochs epochs.\n"

    python3 model/src/main.py --phase "$phase" --epochs $epochs
    if [[ -e model/model_last/model.onnx ]]; then
        echo -e "[*] Moving ONNX Runtime MLP model to backend/src/main/resources/"
        mv model/model_last/model.onnx backend/src/main/resources
        echo -e "[*] Moving ONNX Runtime MinMaxScaler model to backend/src/main/resources/"
        mv model/model_last/scaler.onnx backend/src/main/resources
    fi
else
    echo -e "\n[*] Executing the $phase step...\n"
    python3 model/src/main.py --phase "$phase"
fi
