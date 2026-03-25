package com.example.goldentime.dashboard.repository;

import com.example.goldentime.dashboard.entity.GtOcr;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import org.springframework.data.jpa.repository.Query;
import java.util.Optional;

@Repository
public interface GtOcrRepository extends JpaRepository<GtOcr, Long> {
    Optional<GtOcr> findByGtEvent_GtId(Long gtId);

    @Query("SELECT AVG(o.confidence) FROM GtOcr o")
    Double getAverageConfidence();
}
