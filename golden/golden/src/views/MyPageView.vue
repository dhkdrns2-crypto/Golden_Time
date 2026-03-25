<script setup>
import { ref } from 'vue'
import { useAuth } from '../store/auth'

const { currentUser, updateProfile, addVehicle } = useAuth()

const form = ref({
  name: currentUser.value?.userName ?? '',
  phone: currentUser.value?.phone ?? '',
  email: currentUser.value?.email ?? '',
  address: currentUser.value?.address ?? '',
  newPassword: '',
})

const vehicleForm = ref({
  carNumber: '',
  serialNumber: '',
})

const isAddingVehicle = ref(false)
const msg = ref('')
const msgType = ref('success')

function handleSave() {
  if (!form.value.name.trim()) {
    msg.value = '이름을 입력하세요.'
    msgType.value = 'error'
    return
  }

  const updates = {
    name: form.value.name,
    phone: form.value.phone,
    email: form.value.email,
    address: form.value.address,
  }

  if (form.value.newPassword) {
    if (form.value.newPassword.length < 4) {
      msg.value = '비밀번호는 4자 이상이어야 합니다.'
      msgType.value = 'error'
      return
    }
    updates.password = form.value.newPassword
    form.value.newPassword = ''
  }

  updateProfile(updates)
  msg.value = '정보가 수정되었습니다.'
  msgType.value = 'success'
  setTimeout(() => (msg.value = ''), 3000)
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
            <input v-model="form.name" type="text" class="field-input" />
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
                <td colspan="2" style="text-align: center; color: #94a3b8; padding: 20px;">등록된 차량 정보가 없습니다.</td>
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

/* 기본 정보 필드 */
.info-section {
  margin-bottom: 28px;
}

.field-row {
  padding: 12px 0;
  border-bottom: 1px solid #edf2f7;
}

.field-label {
  display: block;
  font-size: 0.82rem;
  color: #64748b;
  margin-bottom: 4px;
  font-weight: 600;
}

.hint {
  font-size: 0.75rem;
  color: #94a3b8;
  font-weight: 400;
}

.field-input {
  width: 100%;
  border: none;
  outline: none;
  font-size: 0.93rem;
  color: #1e293b;
  background: transparent;
  padding: 4px 0;
  font-family: inherit;
}

.field-input::placeholder {
  color: #cbd5e1;
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
  color: #1e293b;
  margin-bottom: 0;
}

.btn-add-circle {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  border: 1px solid #e2e8f0;
  background: #fff;
  color: #64748b;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-add-circle:hover {
  background: #f8fafc;
  color: #1e293b;
  border-color: #cbd5e1;
}

/* 차량 추가 패널 */
.add-vehicle-panel {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
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
  color: #64748b;
}

.add-input-group input {
  padding: 8px 12px;
  border: 1px solid #e2e8f0;
  border-radius: 4px;
  font-size: 0.88rem;
  outline: none;
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
  border: 1px solid #e2e8f0;
}

.data-table th {
  text-align: left;
  padding: 12px 20px;
  background: #f8fafc;
  color: #4a5568;
  font-weight: 600;
  border-bottom: 1px solid #e2e8f0;
}

.data-table td {
  padding: 12px 20px;
  color: #2d3748;
}

/* 비밀번호 */
.pw-section {
  margin-bottom: 24px;
  padding-bottom: 12px;
  border-bottom: 1px solid #edf2f7;
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
  background: #1a202c;
  color: #fff;
  border: none;
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s;
}

.btn-dark:hover {
  background: #2d3748;
}
</style>
