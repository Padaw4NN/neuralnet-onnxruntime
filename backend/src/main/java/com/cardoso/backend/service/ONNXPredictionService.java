package com.cardoso.backend.service;

import org.springframework.stereotype.Service;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.nio.file.Paths;
import java.util.Collections;
import java.util.Map;


@Service
public class ONNXPredictionService {
    private OrtEnvironment env;
    private OrtSession scalerSession;
    private OrtSession modelSession;

    public ONNXPredictionService() throws OrtException {
        env = OrtEnvironment.getEnvironment();

        String scalerPath = Paths.get("src/main/resources/scaler.onnx").toString();
        String modelPath = Paths.get("src/main/resources/model.onnx").toString();

        scalerSession = env.createSession(scalerPath, new OrtSession.SessionOptions());
        modelSession = env.createSession(modelPath, new OrtSession.SessionOptions());
    }

    private float[] normalizeInput(float[] inputData) throws OrtException {
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, new float[][]{inputData});

        Map<String, OnnxTensor> inputs = Collections.singletonMap("input", inputTensor);
        OrtSession.Result scalerResults = scalerSession.run(inputs);

        float[][] transformedData = (float[][]) scalerResults.get(0).getValue();
        return transformedData[0];
    }

    public int predict(float[] inputData) throws OrtException {
        float[] normalizedData = normalizeInput(inputData);

        OnnxTensor modelInputTensor = OnnxTensor.createTensor(env, new float[][]{normalizedData});
        
        Map<String, OnnxTensor> inputs = Collections.singletonMap("input", modelInputTensor);
        OrtSession.Result results = modelSession.run(inputs);

        float[][] output = (float[][]) results.get(0).getValue();
        int prediction = argmax(output[0]);

        return prediction;
    }

    private static int argmax(float[] predictions) {
        int maxIndex = 0;
        float maxValue = predictions[0];
        for (int i = 1; i < predictions.length; i++) {
            if (predictions[i] > maxValue) {
                maxValue = predictions[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
