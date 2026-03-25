<script setup>
import { ref, computed } from 'vue'
import { useAuth } from '../store/auth'
import { useData } from '../store/data'

const { currentUser, isAdmin } = useAuth()
const { inquiries, addInquiry, incrementInquiryViews, updateInquiry, deleteInquiry } = useData()

const viewMode = ref('list')
const selectedInquiry = ref(null)
const editingAnswer = ref('')
const isEditMode = ref(false)

// 목록 (관리자는 전체, 사용자는 본인 것만)
const visibleInquiries = computed(() => {
  if (isAdmin.value) return inquiries.value
  return inquiries.value.filter((i) => i.authorId === currentUser.value?.id)
})

// 인기글 TOP 3 (조회수 기준, 조회수 > 0인 것만)
const popularIds = computed(() =>
  [...inquiries.value]
    .filter((i) => (i.views || 0) > 0)
    .sort((a, b) => (b.views || 0) - (a.views || 0))
    .slice(0, 3)
    .map((i) => i.id)
)

function popularRank(id) {
  const idx = popularIds.value.indexOf(id)
  return idx >= 0 ? idx + 1 : 0
}

// 글쓰기 폼
const form = ref({ title: '', content: '', image: '' })
const imageFileName = ref('')
const formMsg = ref('')

function openDetail(inq) {
  incrementInquiryViews(inq.id)
  selectedInquiry.value = inq
  editingAnswer.value = inq.answer || ''
  isEditMode.value = false
  viewMode.value = 'detail'
}

function openForm(inq = null) {
  isEditMode.value = !!inq
  form.value = inq
    ? { title: inq.title, content: inq.content, image: inq.image || '' }
    : { title: '', content: '', image: '' }
  imageFileName.value = ''
  formMsg.value = ''
  selectedInquiry.value = inq
  viewMode.value = 'form'
}

function handleImageChange(e) {
  const file = e.target.files?.[0]
  if (!file) return
  imageFileName.value = file.name
  const reader = new FileReader()
  reader.onload = (ev) => { form.value.image = ev.target?.result }
  reader.readAsDataURL(file)
}

function handleSubmit() {
  if (!form.value.title.trim() || !form.value.content.trim()) {
    formMsg.value = '제목과 내용을 모두 입력하세요.'
    return
  }
  if (isEditMode.value && selectedInquiry.value) {
    updateInquiry(selectedInquiry.value.id, {
      title: form.value.title,
      content: form.value.content,
      image: form.value.image,
    })
  } else {
    addInquiry({
      title: form.value.title,
      content: form.value.content,
      image: form.value.image,
      authorId: currentUser.value?.id ?? '',
      authorName: currentUser.value?.name ?? '',
    })
  }
  viewMode.value = 'list'
}

function handleDelete(id) {
  if (confirm('정말 삭제하시겠습니까?')) {
    deleteInquiry(id)
    viewMode.value = 'list'
  }
}

function handleAnswerSave() {
  if (!selectedInquiry.value) return
  updateInquiry(selectedInquiry.value.id, {
    answer: editingAnswer.value,
    status: editingAnswer.value.trim() ? 'answered' : 'pending',
  })
  selectedInquiry.value = {
    ...selectedInquiry.value,
    answer: editingAnswer.value,
    status: editingAnswer.value.trim() ? 'answered' : 'pending',
  }
  alert('답변이 저장되었습니다.')
}

function canEditOrDelete(inq) {
  return inq.authorId === currentUser.value?.id
}
</script>

<template>
  <div class="inquiry">
    <!-- 목록 -->
    <template v-if="viewMode === 'list'">
      <div class="panel">
        <div class="page-header">
          <div>
            <h2 class="page-title">문의게시판</h2>
          </div>
        </div>

        <table v-if="visibleInquiries.length > 0" class="data-table">
          <thead>
            <tr>
              <th class="col-no">#</th>
              <th>제목</th>
              <th>작성자</th>
              <th class="col-status">상태</th>
              <th class="col-views">조회수</th>
              <th class="col-date">등록일</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="(inq, idx) in visibleInquiries"
              :key="inq.id"
              class="clickable-row"
              @click="openDetail(inq)"
            >
              <td>{{ visibleInquiries.length - idx }}</td>
              <td>
                <span v-if="popularRank(inq.id)" class="popular-badge">
                  🔥 인기 {{ popularRank(inq.id) }}위
                </span>
                {{ inq.title }}
              </td>
              <td>{{ inq.authorName }}</td>
              <td>
                <span :class="['badge', inq.status === 'answered' ? 'badge-answered' : 'badge-pending']">
                  {{ inq.status === 'answered' ? '답변완료' : '대기중' }}
                </span>
              </td>
              <td>{{ inq.views || 0 }}</td>
              <td>{{ inq.createdAt }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </template>

    <!-- 상세 -->
    <template v-else-if="viewMode === 'detail' && selectedInquiry">
      <div class="panel">
        <div class="detail-top">
          <button class="btn-back" @click="viewMode = 'list'">← 목록으로</button>
          <div class="header-actions">
            <template v-if="canEditOrDelete(selectedInquiry)">
              <button class="btn-secondary" @click="openForm(selectedInquiry)">수정</button>
              <button class="btn-danger" @click="handleDelete(selectedInquiry.id)">삭제</button>
            </template>
          </div>
        </div>

        <div class="detail-header">
          <span :class="['badge', selectedInquiry.status === 'answered' ? 'badge-answered' : 'badge-pending']">
            {{ selectedInquiry.status === 'answered' ? '답변완료' : '대기중' }}
          </span>
          <h2 class="detail-title">{{ selectedInquiry.title }}</h2>
          <div class="detail-info">
            <span>작성자: {{ selectedInquiry.authorName }}</span>
            <span>등록일: {{ selectedInquiry.createdAt }}</span>
            <span>조회수: {{ selectedInquiry.views || 0 }}</span>
          </div>
        </div>

        <div class="detail-content">{{ selectedInquiry.content }}</div>
        <img v-if="selectedInquiry.image" :src="selectedInquiry.image" class="detail-img" alt="" />

        <!-- 답변 영역 -->
        <div class="answer-section">
          <h3 class="answer-title">답변</h3>
          <template v-if="isAdmin">
            <textarea
              v-model="editingAnswer"
              class="answer-textarea"
              placeholder="답변을 입력하세요"
              rows="4"
            />
            <button class="btn-dark" @click="handleAnswerSave">답변 저장</button>
          </template>
          <template v-else>
            <div v-if="selectedInquiry.answer" class="answer-content">
              {{ selectedInquiry.answer }}
            </div>
            <div v-else class="no-answer">아직 답변이 등록되지 않았습니다.</div>
          </template>
        </div>
      </div>
    </template>

    <!-- 작성/수정 폼 -->
    <template v-else-if="viewMode === 'form'">
      <div class="panel">
        <h2 class="page-title">{{ isEditMode ? '문의 수정' : '문의 작성' }}</h2>

        <div class="write-form">
          <div class="form-group">
            <label>제목 *</label>
            <input v-model="form.title" type="text" placeholder="제목을 입력하세요" />
          </div>

          <div class="form-group">
            <label>이미지 첨부</label>
            <div class="file-row">
              <label class="btn-file">
                파일 선택
                <input type="file" accept="image/*" hidden @change="handleImageChange" />
              </label>
              <span class="file-name">{{ imageFileName || '선택된 파일 없음' }}</span>
            </div>
            <img v-if="form.image" :src="form.image" class="preview-img" alt="미리보기" />
          </div>

          <div class="form-group">
            <label>내용 *</label>
            <textarea v-model="form.content" rows="8" placeholder="내용을 입력하세요" />
          </div>

          <div v-if="formMsg" class="msg-error">{{ formMsg }}</div>

          <div class="form-actions">
            <button class="btn-secondary" @click="viewMode = 'list'">취소</button>
            <button class="btn-dark" @click="handleSubmit">
              {{ isEditMode ? '수정 완료' : '등록' }}
            </button>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.inquiry {
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
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.page-title {
  font-size: 1.25rem;
  font-weight: 700;
  color: #1a202c;
  margin-bottom: 4px;
}

/* 버튼 */
.btn-dark {
  padding: 10px 20px;
  background: #1a202c;
  color: #fff;
  border: none;
  font-size: 0.88rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s;
}

.btn-dark:hover {
  background: #2d3748;
}

.btn-secondary {
  padding: 10px 20px;
  background: #fff;
  border: 1px solid #e2e8f0;
  font-weight: 600;
  cursor: pointer;
}

.btn-secondary:hover {
  background: #f8fafc;
}

/* 테이블 */
.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
}

.data-table th {
  text-align: left;
  padding: 14px 20px;
  background: #f8fafc;
  color: #4a5568;
  font-weight: 600;
  border-bottom: 1px solid #e2e8f0;
}

.data-table td {
  padding: 14px 20px;
  border-bottom: 1px solid #edf2f7;
  color: #2d3748;
}
.data-table tr:last-child td { border-bottom: none; }

.col-no { width: 50px; }
.col-status { width: 90px; }
.col-views { width: 70px; }
.col-date { width: 100px; }

.clickable-row { cursor: pointer; transition: background 0.12s; }
.clickable-row:hover { background: #f9f9f9; }

/* 인기글 배지 */
.popular-badge {
  display: inline-block;
  background: #fff7ed;
  color: #ea580c;
  font-size: 0.7rem;
  font-weight: 700;
  padding: 1px 7px;
  border-radius: 4px;
  border: 1px solid #fed7aa;
  margin-right: 6px;
}

/* 상태 배지 */
.badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 2px;
  font-size: 0.76rem;
  font-weight: 600;
}
.badge-answered { background: #dcfce7; color: #16a34a; }
.badge-pending  { background: #fef3c7; color: #d97706; }

/* 상세 */
.detail-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
.header-actions { display: flex; gap: 8px; }

.detail-header {
  border-bottom: 1px solid #e8eef6;
  padding-bottom: 14px;
  margin-bottom: 16px;
}
.detail-title {
  font-size: 1.15rem;
  font-weight: 700;
  margin: 8px 0;
  color: #1a202c;
}
.detail-info {
  display: flex;
  gap: 16px;
  font-size: 0.8rem;
  color: #8a9bb5;
}
.detail-content {
  min-height: 100px;
  line-height: 1.8;
  white-space: pre-wrap;
  padding: 16px;
  background: #fafafa;
  border-radius: 0;
  border: 1px solid #eeeeee;
  font-size: 0.9rem;
  color: #333;
  margin-bottom: 12px;
}
.detail-img {
  max-width: 100%;
  border-radius: 0;
  margin-bottom: 20px;
  display: block;
}

/* 답변 */
.answer-section {
  margin-top: 24px;
  padding-top: 18px;
  border-top: 1px dashed #dddddd;
}
.answer-title {
  font-size: 0.92rem;
  font-weight: 700;
  margin-bottom: 12px;
  color: #1976d2;
}
.answer-textarea {
  width: 100%;
  padding: 12px;
  border: 1px solid #d1dbe8;
  border-radius: 0;
  font-size: 0.9rem;
  resize: vertical;
  outline: none;
  font-family: inherit;
  margin-bottom: 10px;
  box-sizing: border-box;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.answer-textarea:focus {
  border-color: #1565c0;
  box-shadow: 0 0 0 3px rgba(21, 101, 192, 0.08);
}
.answer-content {
  padding: 14px 16px;
  background: #e3f0fd;
  border-radius: 0;
  line-height: 1.75;
  font-size: 0.9rem;
  white-space: pre-wrap;
  color: #1565c0;
}
.no-answer {
  padding: 20px;
  text-align: center;
  color: #9eaab8;
  font-size: 0.86rem;
  background: #f5f7fa;
  border-radius: 0;
}

/* 작성 폼 */
.write-form {
  display: flex;
  flex-direction: column;
  gap: 16px;
  max-width: 700px;
  margin-top: 8px;
}
.form-group { display: flex; flex-direction: column; gap: 6px; }
.form-group label {
  font-size: 0.82rem;
  color: #6b7a8d;
  font-weight: 500;
}
.form-group input,
.form-group textarea {
  padding: 10px 12px;
  border: 1px solid #d1dbe8;
  border-radius: 0;
  font-size: 0.9rem;
  outline: none;
  font-family: inherit;
  transition: border-color 0.2s, box-shadow 0.2s;
  resize: vertical;
  color: #2d3748;
}
.form-group input:focus,
.form-group textarea:focus {
  border-color: #1565c0;
  box-shadow: 0 0 0 3px rgba(21, 101, 192, 0.08);
}

.file-row { display: flex; align-items: center; gap: 10px; }
.btn-file {
  padding: 6px 14px;
  background: #1565c0;
  color: #fff;
  border-radius: 0;
  font-size: 0.8rem;
  cursor: pointer;
  white-space: nowrap;
  transition: background 0.2s;
}
.btn-file:hover { background: #0d47a1; }
.file-name { font-size: 0.8rem; color: #8a9bb5; }
.preview-img {
  max-width: 200px;
  border-radius: 0;
  margin-top: 8px;
}

.msg-error {
  background: #fee2e2;
  color: #dc2626;
  padding: 10px 14px;
  border-radius: 0;
  font-size: 0.86rem;
}
.form-actions {
  display: flex;
  gap: 10px;
  justify-content: flex-end;
  padding-top: 8px;
  border-top: 1px solid #e8eef6;
}

.empty-msg {
  padding: 40px;
  text-align: center;
  color: #9eaab8;
  font-size: 0.88rem;
  background: #f5f7fa;
  border-radius: 0;
}
</style>
