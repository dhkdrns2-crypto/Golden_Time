package com.example.goldentime.dashboard.repository;

import com.example.goldentime.dashboard.entity.GtEvent;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

import java.time.LocalDateTime;

@Repository
public interface GtEventRepository extends JpaRepository<GtEvent, Long> {

    @Query("SELECT e FROM GtEvent e JOIN FETCH e.vehicle v JOIN FETCH v.user")
    @Override
    List<GtEvent> findAll();

    long countByCreatedAtAfter(LocalDateTime startOfDay);
    long countByCreatedAtAfterAndSentToFireTrue(LocalDateTime startOfDay);
    long countByCreatedAtAfterAndSentToSafetyTrue(LocalDateTime startOfDay);

    List<GtEvent> findTop5ByOrderByCreatedAtDesc();

    @Query("SELECT e.vtIdPath FROM GtEvent e WHERE e.vtIdPath IS NOT NULL")
    List<String> findAllVtIdPaths();
}
