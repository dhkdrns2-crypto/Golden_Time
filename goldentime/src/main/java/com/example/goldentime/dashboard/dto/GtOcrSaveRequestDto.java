package com.example.goldentime.dashboard.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class GtOcrSaveRequestDto {
    private Long gtId;
    private String detectedPlate;
    private Float confidence;

    public Long getGtId() { return gtId; }
    public void setGtId(Long gtId) { this.gtId = gtId; }
    public String getDetectedPlate() { return detectedPlate; }
    public void setDetectedPlate(String detectedPlate) { this.detectedPlate = detectedPlate; }
    public Float getConfidence() { return confidence; }
    public void setConfidence(Float confidence) { this.confidence = confidence; }
}
