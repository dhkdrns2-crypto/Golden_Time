package com.example.goldentime.dashboard.entity;

import java.time.LocalDateTime;

import org.hibernate.annotations.CreationTimestamp;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import jakarta.persistence.OneToOne;
import jakarta.persistence.MapsId;

@Getter
@Setter
@NoArgsConstructor
@Entity
@Table(name = "TB_GT_OCR", schema = "goldentime")
public class GtOcr {

    @Id
    @Column(name = "gt_id")
    private Long gtId;

    @OneToOne(fetch = FetchType.LAZY)
    @MapsId
    @JoinColumn(name = "gt_id")
    private GtEvent gtEvent;

    @Column(name = "detected_plate", length = 20)
    private String detectedPlate;

    @Column(name = "confidence")
    private Float confidence;

    @CreationTimestamp
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    public Long getGtId() { return gtId; }
    public void setGtId(Long gtId) { this.gtId = gtId; }
    public GtEvent getGtEvent() { return gtEvent; }
    public void setGtEvent(GtEvent gtEvent) { this.gtEvent = gtEvent; }
    public String getDetectedPlate() { return detectedPlate; }
    public void setDetectedPlate(String detectedPlate) { this.detectedPlate = detectedPlate; }
    public Float getConfidence() { return confidence; }
    public void setConfidence(Float confidence) { this.confidence = confidence; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
}
