package com.example.goldentime.dashboard.dto;

import com.example.goldentime.dashboard.entity.GtEvent;
import com.example.goldentime.dashboard.entity.GtOcr;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Getter
@Setter
@NoArgsConstructor
public class GtEventResponseDto {
    private Long gtId;
    private Long vehicleId;
    private String carNumber;
    private String serialNumber;
    private String videoPath;
    private String vtIdPath;
    private boolean sentToFire;
    private boolean sentToSafety;
    private LocalDateTime createdAt;
    
    // OCR 정보 추가
    private String detectedPlate;
    private Float confidence;

    public GtEventResponseDto(GtEvent entity) {
        this.gtId = entity.getGtId();
        this.vehicleId = entity.getVehicle().getVehicleId();
        this.carNumber = entity.getVehicle().getCarNumber();
        this.serialNumber = entity.getVehicle().getSerialNumber();
        this.videoPath = entity.getVideoPath();
        this.vtIdPath = entity.getVtIdPath();
        this.sentToFire = entity.isSentToFire();
        this.sentToSafety = entity.isSentToSafety();
        this.createdAt = entity.getCreatedAt();
        
        // OCR 결과 매핑 (있을 경우)
        if (entity.getOcrResult() != null) {
            this.detectedPlate = entity.getOcrResult().getDetectedPlate();
            this.confidence = entity.getOcrResult().getConfidence();
        }
    }
}
