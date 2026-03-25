<script setup>
import { computed, onMounted } from 'vue'
import { useData } from '../store/data'
import { useAuth } from '../store/auth'

const { events, fetchEvents, stats, fetchStats, recentEvents, fetchRecentEvents, eventsByRegion, fetchEventsByRegion } = useData()
const { users, fetchUsers, isAdmin } = useAuth()

onMounted(async () => {
  await fetchEvents()
  await fetchStats()
  await fetchRecentEvents()
  await fetchEventsByRegion()
  if (isAdmin.value) {
    await fetchUsers()
  }
})

const totalEvents = computed(() => events.value.length)
const unsentCount = computed(() => 
  events.value.filter(ev => !ev.sentToFire && !ev.sentToSafety).length
)

// 자원 활용도 계산 (기기번호 기준)
// 전체 기기 수: 등록된 모든 차량의 기기번호 총합
const totalDevices = computed(() => {
  if (!isAdmin.value) return 0
  return users.value.reduce((acc, user) => acc + (user.vehicles ? user.vehicles.length : 0), 0)
})

// 사용 중인 기기 수: 한 번이라도 신고 이벤트를 발생시킨 고유 기기번호 수
const usedDevices = computed(() => {
  const uniqueSerials = new Set(events.value.map(ev => ev.serialNumber).filter(s => s))
  return uniqueSerials.size
})

const resourcePercent = computed(() => {
  if (totalDevices.value === 0) return 0
  return Math.round((usedDevices.value / totalDevices.value) * 100)
})

// 도넛 차트: r=38, 둘레 = 2π×38 ≈ 238.8
const donutArc = computed(() => resourcePercent.value * 2.388)

// 번호판 인식 정확도 (평균 confidence)
const averageAccuracy = computed(() => {
  const val = stats.value?.averageConfidence || 0
  return (val * 100).toFixed(1)
})

// 오늘의 통계 데이터
const todayTotal = computed(() => stats.value?.totalEventsToday || 0)
const todayFire = computed(() => stats.value?.sentToFireToday || 0)
const todaySafety = computed(() => stats.value?.sentToSafetyToday || 0)

const topRegions = computed(() => {
  const source = eventsByRegion.value || {}
  return Object.entries(source)
    .map(([name, count]) => ({ name, count: Number(count) || 0 }))
    .filter(item => item.count > 0)
    .sort((a, b) => b.count - a.count)
    .slice(0, 6)
})

const maxRegionCount = computed(() => {
  return topRegions.value.length ? topRegions.value[0].count : 1
})

</script>

<template>
  <div class="main-view">
    <div class="panel">
      <div class="page-header">
        <div>
          <h2 class="page-title">대시보드</h2>
          <p class="page-subtitle">시스템의 실시간 활성 사건 및 자원 활용도를 파악하세요.</p>
        </div>
      </div>

      <div class="content-body">
        <!-- 대시보드 메인 그리드 -->
        <div class="dashboard-main-grid">
          <!-- 상단 요약 카드들 -->
          <div class="summary-card total">
            <div class="summary-icon">🚨</div>
            <div class="summary-info">
              <span class="summary-label">오늘의 탐지</span>
              <span class="summary-value">{{ todayTotal }}건</span>
            </div>
          </div>
          <div class="summary-card fire">
            <div class="summary-icon">🚒</div>
            <div class="summary-info">
              <span class="summary-label">소방청 전송</span>
              <span class="summary-value">{{ todayFire }}건</span>
            </div>
          </div>
          <div class="summary-card safety">
            <div class="summary-icon">🛡️</div>
            <div class="summary-info">
              <span class="summary-label">안전신문고 전송</span>
              <span class="summary-value">{{ todaySafety }}건</span>
            </div>
          </div>

          <!-- 지역별 사건 발생 현황 (리스트형으로 재구성) -->
          <section class="stat-card region-card" aria-label="지역별 사건 발생 현황">
            <div class="region-card-head">
              <p class="card-label">지역별 사건 발생 현황</p>
            </div>
            <div class="region-card-body">
              <div v-if="topRegions.length === 0" class="region-empty">
                집계된 지역 데이터가 없습니다.
              </div>
              <ul v-else class="region-list">
                <li v-for="(region, idx) in topRegions" :key="region.name" class="region-item">
                  <div class="region-meta">
                    <span class="region-rank">{{ idx + 1 }}</span>
                    <span class="region-name">{{ region.name }}</span>
                    <span class="region-count">{{ region.count }}건</span>
                  </div>
                  <div class="region-bar-bg">
                    <div
                      class="region-bar-fill"
                      :style="{ width: `${(region.count / maxRegionCount) * 100}%` }"
                    ></div>
                  </div>
                </li>
              </ul>
            </div>
            <p class="card-desc">상위 지역 기준 실시간 발생 분포</p>
          </section>

          <!-- 하단 상세 지표 카드들 -->
          <div class="stat-card realtime-card">
            <p class="card-label">실시간 활성 사건</p>
            <div class="big-value">
              {{ totalEvents }}건
              <span class="change up">
                <span class="dot red"></span> 미전송 {{ unsentCount }}건
              </span>
            </div>
            <p class="card-desc">현재 접수된 전체 신고 사건 현황</p>
          </div>

          <div class="stat-card resource-card">
            <p class="card-label">자원 활용도 (기기 가동률)</p>
            <div class="donut-wrap">
              <svg viewBox="0 0 100 100" class="donut-svg">
                <circle class="donut-track" cx="50" cy="50" r="38" />
                <circle
                  class="donut-arc"
                  cx="50" cy="50" r="38"
                  transform="rotate(-90 50 50)"
                  :stroke-dasharray="`${donutArc} 238.8`"
                />
              </svg>
              <div class="donut-center">
                <span class="donut-val">{{ resourcePercent }}%</span>
                <span class="donut-sub">{{ usedDevices }}/{{ totalDevices }} 유닛</span>
              </div>
            </div>
            <p class="card-desc">전체 등록 기기 중 이벤트 발생 기기 비율</p>
          </div>

          <div class="stat-card accuracy-card">
            <p class="card-label">번호판 인식 모델 정확도</p>
            <div class="accuracy-content">
              <div class="accuracy-value">
                <span class="val-num">{{ averageAccuracy }}</span>
                <span class="val-unit">%</span>
              </div>
              <div class="accuracy-bar-bg">
                <div class="accuracy-bar-fill" :style="{ width: averageAccuracy + '%' }"></div>
              </div>
            </div>
            <p class="card-desc">OCR 분석 결과의 평균 신뢰도 지수</p>
          </div>
        </div>

        <!-- 최신 신고 목록 -->
        <div class="recent-events-card">
          <p class="card-label">최신 신고 목록 (5건)</p>
          <table class="events-table">
            <thead>
              <tr>
                <th>차량 번호</th>
                <th>인식된 번호판</th>
                <th>발생 시각</th>
                <th>신고 상태</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="event in recentEvents" :key="event.gtId">
                <td>{{ event.carNumber }}</td>
                <td>{{ event.detectedPlate || '-' }}</td>
                <td>{{ new Date(event.createdAt).toLocaleString() }}</td>
                <td>
                  <span v-if="event.sentToFire" class="status-tag fire">소방청</span>
                  <span v-else-if="event.sentToSafety" class="status-tag safety">안전신문고</span>
                  <span v-else class="status-tag pending">미전송</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.main-view {
  padding: 24px;
  background: #f8fafc;
  min-height: 100%;
}

.panel {
  background: #fff;
  border: 1px solid #e2e8f0;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.page-header {
  padding: 24px 28px;
  border-bottom: 1px solid #edf2f7;
}

.page-title {
  font-size: 1.25rem;
  font-weight: 700;
  color: #1a202c;
  margin-bottom: 4px;
}

.page-subtitle {
  font-size: 0.88rem;
  color: #718096;
}

.content-body {
  padding: 28px;
}

.dashboard-main-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  grid-template-rows: 90px minmax(0, 1fr);
  gap: 20px;
  margin-bottom: 24px;
  height: 420px;
}

.grid-col {
  display: flex;
  flex-direction: column;
  gap: 20px;
  height: 100%;
}

.summary-card {
  background: white;
  padding: 20px;
  border: 1px solid #eef2f7;
  display: flex;
  align-items: center;
  gap: 16px;
  transition: all 0.2s ease;
  height: 100%;
  box-sizing: border-box;
}

.summary-icon {
  font-size: 28px;
  width: 52px;
  height: 52px;
  background: #f8fafc;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid #f1f5f9;
}

.summary-info {
  display: flex;
  flex-direction: column;
}

.summary-label {
  font-size: 0.875rem;
  color: #64748b;
  font-weight: 500;
}

.summary-value {
  font-size: 1.25rem;
  font-weight: 700;
  color: #1e293b;
}

/* 최신 신고 목록 */
.recent-events-card {
  background: white;
  padding: 20px;
  border: 1px solid #eef2f7;
  margin-top: 24px;
}

.events-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 12px;
}

.events-table th, .events-table td {
  padding: 12px;
  text-align: left;
  border-bottom: 1px solid #f1f5f9;
  font-size: 0.9rem;
}

.events-table th {
  color: #64748b;
  font-weight: 600;
  background: #f8fafc;
}

.status-tag {
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 600;
}
.status-tag.fire { background: #fee2e2; color: #ef4444; }
.status-tag.safety { background: #d1fae5; color: #10b981; }
.status-tag.pending { background: #f1f5f9; color: #64748b; }

.stat-card {
  background: #fff;
  padding: 20px;
  border: 1px solid #eef2f7;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  transition: all 0.2s ease;
  min-height: 0; /* flex 자식의 overflow 방지 */
  box-sizing: border-box;
  overflow: hidden;
}

.region-card {
  grid-column: 4;
  grid-row: 1 / 3;
  min-height: 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.region-card-head {
  flex-shrink: 0;
}

.region-card-body {
  flex: 1;
  min-height: 0;
}

.region-empty {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #94a3b8;
  font-size: 0.85rem;
}

.region-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.region-item {
  padding: 10px 12px;
  border: 1px solid #edf2f7;
  background: #f8fafc;
}

.region-meta {
  display: grid;
  grid-template-columns: 20px 1fr auto;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.region-rank {
  font-size: 0.78rem;
  font-weight: 800;
  color: #475569;
}

.region-name {
  font-size: 0.85rem;
  font-weight: 700;
  color: #1e293b;
}

.region-count {
  font-size: 0.8rem;
  color: #475569;
  font-weight: 700;
}

.region-bar-bg {
  width: 100%;
  height: 8px;
  background: #e2e8f0;
  overflow: hidden;
}

.region-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #2563eb, #60a5fa);
}

.card-label {
  font-size: 0.82rem;
  color: #64748b;
  font-weight: 600;
  margin-bottom: 12px;
}

.status-value {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}
.status-dot.green { background: #10b981; }

.status-text {
  font-size: 1.1rem;
  font-weight: 700;
  color: #1e293b;
}

.big-value {
  font-size: 1.5rem;
  font-weight: 800;
  color: #1e293b;
  display: flex; /* correct line */
  align-items: baseline;
  gap: 8px;
}

.change {
  font-size: 0.78rem;
  font-weight: 600;
  padding: 2px 6px;
  border-radius: 4px;
}
.change.up { color: #ef4444; }
.change.down { color: #3b82f6; }

.accuracy-content {
  margin: 10px 0;
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.accuracy-value {
  margin-bottom: 12px;
  display: flex;
  align-items: baseline;
  gap: 4px;
}

.accuracy-value .val-num {
  font-size: 2rem;
  font-weight: 800;
  color: #1e293b;
}

.accuracy-value .val-unit {
  font-size: 1rem;
  font-weight: 600;
  color: #64748b;
}

.accuracy-bar-bg {
  width: 100%;
  height: 8px;
  background: #f1f5f9;
  border-radius: 4px;
  overflow: hidden;
}

.accuracy-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #3b82f6, #60a5fa);
  border-radius: 4px;
  transition: width 0.6s ease-out;
}

.card-desc {
  font-size: 0.72rem;
  color: #94a3b8;
  margin-top: 8px;
}

/* 자원 활용도 */
.row-2 {
  grid-template-columns: 1fr 1fr 1.5fr;
}

.donut-wrap {
  position: relative;
  width: 140px;
  height: 140px;
  margin: 0 auto;
}

.donut-track {
  fill: none;
  stroke: #f1f5f9;
  stroke-width: 8;
}

.donut-arc {
  fill: none;
  stroke: #1976d2;
  stroke-width: 8;
  stroke-linecap: round;
  transition: stroke-dasharray 0.5s ease;
}

.donut-center {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.summary-icon {
  font-size: 28px;
  width: 52px;
  height: 52px;
  background: #f8fafc;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid #f1f5f9;
}

.summary-info {
  display: flex;
  flex-direction: column;
}

.summary-label {
  font-size: 0.875rem;
  color: #64748b;
  font-weight: 500;
}

.summary-value {
  font-size: 1.25rem;
  font-weight: 700;
  color: #1e293b;
}

.donut-val {
  font-size: 1.4rem;
  font-weight: 800;
  color: #1e293b;
}

.donut-sub {
  font-size: 0.68rem;
  color: #64748b;
}

/* 바 차트 */
.bar-chart-area {
  display: flex;
  align-items: flex-end;
  justify-content: center;
  gap: 40px;
  height: 140px;
}

.bar-item {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.bar-col {
  width: 40px;
  border-radius: 4px 4px 0 0;
}
.bar-col.success { background: #10b981; }
.bar-col.fail { background: #ef4444; }

.bar-label {
  font-size: 0.75rem;
  color: #64748b;
  margin-top: 8px;
}

.bar-val {
  font-size: 0.88rem;
  font-weight: 700;
  color: #1e293b;
}

/* 핫스팟 */
.hotspot-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.hotspot-item {
  display: flex;
  align-items: center;
  padding: 10px 14px;
  background: #f8fafc;
  border-radius: 4px;
  font-size: 0.84rem;
}

.rank {
  font-weight: 800;
  color: #64748b;
  width: 24px;
}

.area {
  flex: 1;
  font-weight: 600;
  color: #1e293b;
}

.count {
  margin: 0 12px;
  color: #1976d2;
  font-weight: 700;
}

.level {
  font-size: 0.72rem;
  padding: 2px 8px;
  background: #e2e8f0;
  color: #475569;
  border-radius: 10px;
}
.level.high {
  background: #fee2e2;
  color: #b91c1c;
}

/* 전국 지도 */
.map-row {
  margin-top: 24px;
}

.map-card {
  padding: 20px;
}

.korea-map-wrap {
  width: 100%;
  max-width: 400px;
  margin: 0 auto;
  display: flex;
  justify-content: center;
}

.korea-svg {
  width: 100%;
  height: auto;
}

.korea-land {
  fill: #f1f5f9;
  stroke: #cbd5e1;
  stroke-width: 1.5;
}

.map-label {
  font-size: 12px;
  font-weight: 800;
  fill: #1e293b;
}

.map-count {
  font-size: 11px;
  font-weight: 600;
  fill: #dc2626;
}

.map-label-sm {
  font-size: 10px;
  font-weight: 600;
  fill: #64748b;
}
</style>
