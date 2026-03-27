<script setup>
import { ref, watch } from 'vue'
import { useAuth } from '../store/auth'

const { currentUser, updateProfile, addVehicle } = useAuth()

const form = ref({
  userName: '',
  phone: '',
  email: '',
  address: '',
  newPassword: '',
})

// 마운트 시점에는 currentUser 가 아직 없을 수 있음 → 로드되면 폼에 반영
watch(
  currentUser,
  (u) => {
    if (!u) return
    form.value.userName = u.userName ?? ''
    form.value.phone = u.phone ?? ''
    form.value.email = u.email ?? ''
    form.value.address = u.address ?? ''
  },
  { immediate: true },
)

const vehicleForm = ref({
  carNumber: '',
  serialNumber: '',
})

const isAddingVehicle = ref(false)
const msg = ref('')
const msgType = ref('success')

async function handleSave() {
  if (!form.value.userName.trim()) {
    msg.value = '이름을 입력하세요.'
    msgType.value = 'error'
    return
  }

  const updates = {
    userName: form.value.userName.trim(),
    phone: form.value.phone?.trim() ?? '',
    email: form.value.email?.trim() ?? '',
    address: form.value.address?.trim() ?? '',
  }

  if (form.value.newPassword) {
    if (form.value.newPassword.length < 4) {
      msg.value = '비밀번호는 4자 이상이어야 합니다.'
      msgType.value = 'error'
      return
    }
    updates.password = form.value.newPassword
  }

  const res = await updateProfile(updates)
  if (res?.success) {
    form.value.newPassword = ''
    msg.value = '정보가 수정되었습니다.'
    msgType.value = 'success'
  } else {
    msg.value = res?.message || '정보 수정에 실패했습니다. 다시 시도해 주세요.'
    msgType.value = 'error'
  }
  setTimeout(() => (msg.value = ''), 5000)
}

async function handleAddVehicle() {
  if (!vehicleForm.value.carNumber.trim() || !vehicleForm.value.serialNumber.trim()) {
    msg.value = '차량 번호와 단말기 시리얼을 모두 입력하세요.'
    msgType.value = 'error'
    return
  }

  const res = await addVehicle(vehicleForm.value.carNumber, vehicleForm.value.serialNumber)
  if (res.success) {
    msg.value = '차량이 추가되었습니다.'
    msgType.value = 'success'
    vehicleForm.value.carNumber = ''
    vehicleForm.value.serialNumber = ''
    isAddingVehicle.value = false
  } else {
    msg.value = '차량 추가에 실패했습니다.'
    msgType.value = 'error'
  }
  setTimeout(() => (msg.value = ''), 3000)
}
</script>

<template>
  <div class="mypage">
    <div class="panel">
      <div class="page-header">
        <div>
          <h2 class="page-title">마이페이지</h2>
          <p class="page-subtitle">회원님의 개인 정보를 확인하고 수정할 수 있습니다.</p>
        </div>
      </div>

      <div class="content-body">
        <!-- 기본 정보 -->
        <div class="info-section">
          <div class="field-row">
            <span class="field-label">이름</span>
            <input v-model="form.userName" type="text" class="field-input" />
          </div>
          <div class="field-row">
            <span class="field-label">전화번호</span>
            <input v-model="form.phone" type="tel" class="field-input" placeholder="010-0000-0000" />
          </div>
          <div class="field-row">
            <span class="field-label">이메일</span>
            <input v-model="form.email" type="email" class="field-input" />
          </div>
          <div class="field-row">
            <span class="field-label">주소</span>
            <input v-model="form.address" type="text" class="field-input" />
          </div>
        </div>

        <!-- 차량 정보 -->
        <div class="vehicle-section">
          <div class="section-header">
            <p class="section-title">내 차량 정보</p>
            <button class="btn-add-circle" @click="isAddingVehicle = !isAddingVehicle" title="차량 추가">
              <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14" /></svg>
            </button>
          </div>

          <!-- 차량 추가 폼 -->
          <div v-if="isAddingVehicle" class="add-vehicle-panel">
            <div class="add-form-row">
              <div class="add-input-group">
                <label>내 차량 번호</label>
                <input v-model="vehicleForm.carNumber" type="text" placeholder="예: 12가 3456" />
              </div>
              <div class="add-input-group">
                <label>단말기 시리얼</label>
                <input v-model="vehicleForm.serialNumber" type="text" placeholder="시리얼 번호 입력" />
              </div>
              <div class="add-btn-group">
                <button class="btn-dark sm" @click="handleAddVehicle">등록</button>
                <button class="btn-cancel sm" @click="isAddingVehicle = false">취소</button>
              </div>
            </div>
          </div>

          <table class="data-table">
            <thead>
              <tr>
                <th>내 차량 번호</th>
                <th>단말기 시리얼</th>
              </tr>
            </thead>
            <tbody>
              <tr v-if="currentUser?.vehicles && currentUser.vehicles.length > 0" v-for="v in currentUser.vehicles" :key="v.vehicleId">
                <td>{{ v.carNumber || '-' }}</td>
                <td>{{ v.serialNumber || '-' }}</td>
              </tr>
              <tr v-else>
                <td colspan="2" class="empty-vehicles">등록된 차량 정보가 없습니다.</td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- 비밀번호 변경 -->
        <div class="pw-section">
          <p class="field-label">비밀번호 변경 <span class="hint">(필요한 경우에만 입력)</span></p>
          <input
            v-model="form.newPassword"
            type="password"
            class="field-input"
            placeholder="새 비밀번호"
          />
        </div>

        <!-- 메시지 -->
        <div v-if="msg" :class="['msg-box', `msg-${msgType}`]">{{ msg }}</div>

        <!-- 저장 버튼 -->
        <div class="btn-area">
          <button class="btn-dark" @click="handleSave">정보 수정하기</button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.mypage {
  padding: 24px;
  background: var(--page-bg);
  min-height: 100%;
}

.panel {
  background: var(--bg-card);
  border: 1px solid var(--border-solid);
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.page-header {
  padding: 24px 28px;
  border-bottom: 1px solid var(--border-solid);
}

.page-title {
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--text-h);
  margin-bottom: 4px;
}

.page-subtitle {
  font-size: 0.88rem;
  color: var(--text-muted);
}

.content-body {
  padding: 28px;
}

/* 기본 정보 필드 */
.info-section {
  margin-bottom: 28px;
}

.field-row {
  padding: 12px 0;
  border-bottom: 1px solid var(--border-solid);
}

.field-label {
  display: block;
  font-size: 0.82rem;
  color: var(--text-muted);
  margin-bottom: 4px;
  font-weight: 600;
}

.hint {
  font-size: 0.75rem;
  color: var(--text-light);
  font-weight: 400;
}

.field-input {
  width: 100%;
  border: none;
  outline: none;
  font-size: 0.93rem;
  color: var(--text);
  background: transparent;
  padding: 4px 0;
  font-family: inherit;
}

.field-input::placeholder {
  color: rgba(148, 163, 184, 0.8);
}

/* 차량 정보 */
.vehicle-section {
  margin-bottom: 32px;
}

.section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.section-title {
  font-size: 0.9rem;
  font-weight: 700;
  color: var(--text-h);
  margin-bottom: 0;
}

.btn-add-circle {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  border: 1px solid var(--border-solid);
  background: var(--bg-card);
  color: var(--text-muted);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-add-circle:hover {
  background: var(--surface-2);
  color: var(--text-h);
  border-color: var(--border-solid);
}

/* 차량 추가 패널 */
.add-vehicle-panel {
  background: var(--surface-2);
  border: 1px solid var(--border-solid);
  padding: 16px;
  margin-bottom: 16px;
  border-radius: 4px;
}

.add-form-row {
  display: flex;
  gap: 16px;
  align-items: flex-end;
}

.add-input-group {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.add-input-group label {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--text-muted);
}

.add-input-group input {
  padding: 8px 12px;
  border: 1px solid var(--border-solid);
  border-radius: 4px;
  font-size: 0.88rem;
  outline: none;
  background: var(--bg-card);
  color: var(--text);
}

.add-input-group input:focus {
  border-color: #3b82f6;
}

.add-btn-group {
  display: flex;
  gap: 8px;
}

.btn-dark.sm, .btn-cancel.sm {
  padding: 8px 16px;
  font-size: 0.8rem;
}

/* 테이블 공통 스타일 적용 */
.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
  border: 1px solid var(--border-solid);
}

.data-table th {
  text-align: left;
  padding: 12px 20px;
  background: var(--surface-2);
  color: var(--text-muted);
  font-weight: 600;
  border-bottom: 1px solid var(--border-solid);
}

.data-table td {
  padding: 12px 20px;
  color: var(--text);
}

.empty-vehicles {
  text-align: center;
  color: var(--text-muted);
  padding: 20px;
}

/* 비밀번호 */
.pw-section {
  margin-bottom: 24px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border-solid);
}

/* 메시지 */
.msg-box {
  padding: 10px 14px;
  border-radius: 4px;
  font-size: 0.86rem;
  margin-bottom: 16px;
}

.msg-success {
  background: #f0fdf4;
  color: #166534;
  border: 1px solid #bbf7d0;
}

.msg-error {
  background: #fef2f2;
  color: #991b1b;
  border: 1px solid #fecaca;
}

/* 버튼 */
.btn-area {
  display: flex;
  justify-content: flex-end;
}

.btn-dark {
  padding: 10px 24px;
  background: var(--navy);
  color: #fff;
  border: none;
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s;
}

.btn-dark:hover {
  background: var(--btn-hover-bg);
}
</style>
