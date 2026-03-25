package com.example.goldentime.dashboard.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.web.multipart.MultipartFile;

import java.math.BigDecimal;

@Getter
@Setter
@NoArgsConstructor
public class GtEventSaveRequestDto {
    private Long vehicleId;
    private MultipartFile videoFile;
    private String vtIdPath;

    public Long getVehicleId() { return vehicleId; }
    public void setVehicleId(Long vehicleId) { this.vehicleId = vehicleId; }
    public MultipartFile getVideoFile() { return videoFile; }
    public void setVideoFile(MultipartFile videoFile) { this.videoFile = videoFile; }
    public String getVtIdPath() { return vtIdPath; }
    public void setVtIdPath(String vtIdPath) { this.vtIdPath = vtIdPath; }
}
