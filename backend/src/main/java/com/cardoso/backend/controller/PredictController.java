package com.cardoso.backend.controller;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.cardoso.backend.model.PredictRequest;
import com.cardoso.backend.model.PredictResponse;
import com.cardoso.backend.service.ONNXPredictionService;

@RestController
@RequestMapping("/")
public class PredictController {
    private final ONNXPredictionService predictionService;
    
    public PredictController(ONNXPredictionService predictionService) {
        this.predictionService = predictionService;
    }

    @PostMapping("/predict")
    public PredictResponse predict(@RequestBody PredictRequest request) throws Exception {
        float[] inputData = request.toArray();
        int prediction = predictionService.predict(inputData);
        return new PredictResponse(prediction);
    }
}
