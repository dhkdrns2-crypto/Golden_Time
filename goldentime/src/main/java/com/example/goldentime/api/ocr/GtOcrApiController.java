package com.example.goldentime.api.ocr;

import com.example.goldentime.dashboard.dto.GtOcrSaveRequestDto;
import com.example.goldentime.dashboard.entity.GtEvent;
import com.example.goldentime.dashboard.entity.GtOcr;
import com.example.goldentime.dashboard.repository.GtEventRepository;
import com.example.goldentime.dashboard.repository.GtOcrRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/ocr")
@RequiredArgsConstructor
public class GtOcrApiController {

    private final GtOcrRepository gtOcrRepository;
    private final GtEventRepository gtEventRepository;

    @PostMapping("/save")
    public ResponseEntity<Long> saveOcrResult(@RequestBody GtOcrSaveRequestDto requestDto) {
        GtEvent gtEvent = gtEventRepository.findById(requestDto.getGtId())
                .orElseThrow(() -> new IllegalArgumentException("이벤트를 찾을 수 없습니다. ID: " + requestDto.getGtId()));

        GtOcr gtOcr = new GtOcr();
        gtOcr.setGtEvent(gtEvent);
        gtOcr.setDetectedPlate(requestDto.getDetectedPlate());
        gtOcr.setConfidence(requestDto.getConfidence());

        GtOcr saved = gtOcrRepository.save(gtOcr);
        return ResponseEntity.ok(saved.getGtId());
    }
}
