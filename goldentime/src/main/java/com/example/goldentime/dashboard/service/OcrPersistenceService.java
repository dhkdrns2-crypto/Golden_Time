package com.example.goldentime.dashboard.service;

import com.example.goldentime.dashboard.entity.GtEvent;
import com.example.goldentime.dashboard.entity.GtOcr;
import com.example.goldentime.dashboard.repository.GtEventRepository;
import com.example.goldentime.dashboard.repository.GtOcrRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Map;

@Service
@RequiredArgsConstructor
public class OcrPersistenceService {

    private final GtEventRepository gtEventRepository;
    private final GtOcrRepository gtOcrRepository;

    @Transactional
    public void saveOcrResult(Long eventId, Map<String, Object> ocrData) {
        GtEvent event = gtEventRepository.findById(eventId)
                .orElseThrow(() -> new IllegalArgumentException("이벤트를 찾을 수 없습니다. ID: " + eventId));

        GtOcr ocrResult = new GtOcr();
        ocrResult.setGtEvent(event);
        ocrResult.setDetectedPlate((String) ocrData.get("detected_plate"));
        ocrResult.setConfidence(((Number) ocrData.get("confidence")).floatValue());
        gtOcrRepository.save(ocrResult);
        System.out.println("Successfully saved OCR result for event " + eventId + " in a new transaction.");
    }
}
