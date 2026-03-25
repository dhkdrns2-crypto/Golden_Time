<script setup>
import { computed } from 'vue'

const props = defineProps({
  eventCounts: {
    type: Object,
    default: () => ({})
  }
})

// 각 지역별 중심 좌표 (대략적인 위치)
const regions = [
  { id: 'seoul', name: '서울특별시', cx: 150, cy: 90 },
  { id: 'busan', name: '부산광역시', cx: 250, cy: 320 },
  { id: 'daegu', name: '대구광역시', cx: 230, cy: 250 },
  { id: 'incheon', name: '인천광역시', cx: 120, cy: 110 },
  { id: 'gwangju', name: '광주광역시', cx: 120, cy: 290 },
  { id: 'daejeon', name: '대전광역시', cx: 160, cy: 200 },
  { id: 'ulsan', name: '울산광역시', cx: 260, cy: 270 },
  { id: 'sejong', name: '세종특별자치시', cx: 150, cy: 170 },
  { id: 'gyeonggi', name: '경기도', cx: 160, cy: 70 },
  { id: 'gangwon', name: '강원도', cx: 220, cy: 50 },
  { id: 'chungbuk', name: '충청북도', cx: 200, cy: 120 },
  { id: 'chungnam', name: '충청남도', cx: 130, cy: 150 },
  { id: 'jeonbuk', name: '전라북도', cx: 140, cy: 230 },
  { id: 'jeonnam', name: '전라남도', cx: 110, cy: 320 },
  { id: 'gyeongbuk', name: '경상북도', cx: 240, cy: 170 },
  { id: 'gyeongnam', name: '경상남도', cx: 200, cy: 290 },
  { id: 'jeju', name: '제주특별자치도', cx: 100, cy: 390 }
]

// API에서 받은 데이터와 매핑
const mapData = computed(() => {
  return regions.map(region => {
    // API 응답 키(예: "전라남도")와 매칭
    const count = props.eventCounts[region.name] || 0
    return {
      ...region,
      count,
      // 건수에 따라 점 크기 조절 (최소 4, 최대 15)
      radius: count > 0 ? Math.min(15, 4 + count * 2) : 0
    }
  })
})
</script>

<template>
  <div class="korea-map-wrap">
    <svg viewBox="70 0 220 410" preserveAspectRatio="xMidYMid meet" class="korea-svg">
      <!-- 한반도 대략적인 윤곽선 (단순화된 형태) -->
      <path class="korea-land" d="M120,20 L180,0 L240,30 L260,90 L280,170 L260,270 L280,320 L220,350 L150,330 L100,350 L80,290 L100,220 L80,150 L100,70 Z" />
      <!-- 제주도 -->
      <ellipse class="korea-land" cx="100" cy="390" rx="20" ry="10" />

      <!-- 각 지역별 점 및 라벨 -->
      <g v-for="region in mapData" :key="region.id">
        <circle 
          v-if="region.count > 0"
          :cx="region.cx" 
          :cy="region.cy" 
          :r="region.radius" 
          class="event-dot" 
        />
        <text 
          v-if="region.count > 0"
          :x="region.cx" 
          :y="region.cy - region.radius - 5" 
          class="map-label"
        >
          {{ region.name.substring(0, 2) }} ({{ region.count }})
        </text>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.korea-map-wrap {
  width: 100%;
  flex: 1; /* 부모(stat-card)의 남은 공간 차지 */
  display: flex;
  justify-content: center;
  align-items: center;
  background: #f8fafc;
  border-radius: 8px;
  padding: 4px;
  min-height: 0; /* overflow 방지 */
}

.korea-svg {
  width: 100%;
  height: 100%;
}

.korea-land {
  fill: #e2e8f0;
  stroke: #cbd5e1;
  stroke-width: 2;
}

.event-dot {
  fill: #ef4444;
  opacity: 0.8;
}

.map-label {
  font-size: 12px;
  font-weight: 700;
  fill: #1e293b;
  text-anchor: middle;
}
</style>
