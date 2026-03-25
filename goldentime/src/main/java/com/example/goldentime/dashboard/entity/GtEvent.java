package com.example.goldentime.dashboard.entity;

import jakarta.persistence.CascadeType;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.OneToOne;
import jakarta.persistence.Table;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import jakarta.persistence.OneToMany;
import java.util.ArrayList;
import java.util.List;
import java.time.LocalDateTime;
import org.hibernate.annotations.CreationTimestamp;
import com.example.goldentime.user.entity.UserVehicle;

@Getter
@Setter
@NoArgsConstructor
@Entity
@Table(name = "TB_GT_EVENT", schema = "goldentime")
public class GtEvent {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "gt_id")
    private Long gtId;

    @OneToOne(mappedBy = "gtEvent", fetch = FetchType.LAZY, cascade = CascadeType.ALL, orphanRemoval = true)
    private GtOcr ocrResult;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "vehicle_id", nullable = false)
    private UserVehicle vehicle;

    @Column(name = "video_path", length = 500, unique = true)
    private String videoPath;

    @Column(name = "vt_id_path", length = 500)
    private String vtIdPath;

    @Column(name = "sent_to_fire")
    private boolean sentToFire;

    @Column(name = "sent_to_safety")
    private boolean sentToSafety;

    @CreationTimestamp
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    public Long getGtId() { return gtId; }
    public void setGtId(Long gtId) { this.gtId = gtId; }
    public GtOcr getOcrResult() { return ocrResult; }
    public void setOcrResult(GtOcr ocrResult) { this.ocrResult = ocrResult; }
    public UserVehicle getVehicle() { return vehicle; }
    public void setVehicle(UserVehicle vehicle) { this.vehicle = vehicle; }
    public String getVideoPath() { return videoPath; }
    public void setVideoPath(String videoPath) { this.videoPath = videoPath; }
    public String getVtIdPath() { return vtIdPath; }
    public void setVtIdPath(String vtIdPath) { this.vtIdPath = vtIdPath; }
    public boolean isSentToFire() { return sentToFire; }
    public void setSentToFire(boolean sentToFire) { this.sentToFire = sentToFire; }
    public boolean isSentToSafety() { return sentToSafety; }
    public void setSentToSafety(boolean sentToSafety) { this.sentToSafety = sentToSafety; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
}
