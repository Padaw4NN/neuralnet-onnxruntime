package com.cardoso.backend.model;

import lombok.Data;

@Data
public class PredictRequest {
    private float nitrogen;
    private float phosphorus;
    private float potassium;
    private float temperature;
    private float humidity;
    private float ph;
    private float rainfall;

    public float[] toArray() {
        return new float[] {
            nitrogen,
            phosphorus,
            potassium,
            temperature,
            humidity,
            ph,
            rainfall
        };
    }
}
