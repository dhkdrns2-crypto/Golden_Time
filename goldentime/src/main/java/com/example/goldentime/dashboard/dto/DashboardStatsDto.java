package com.example.goldentime.dashboard.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class DashboardStatsDto {
    private Double averageConfidence;
    private Long totalEventsToday;
    private Long sentToFireToday;
    private Long sentToSafetyToday;
}
