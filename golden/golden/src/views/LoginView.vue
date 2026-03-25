<script setup>
import { ref, onMounted } from "vue";
import { useRouter, useRoute } from "vue-router";
import { useAuth } from "../store/auth";

const router = useRouter();
const route = useRoute();
const { login, isAdmin } = useAuth();

const username = ref("");
const password = ref("");
const notice = ref("");
const isError = ref(false);

onMounted(() => {
  if (route.query.logout === "1") {
    notice.value = "로그아웃되었습니다.";
    isError.value = false;
  }
});

async function handleLogin() {
  const result = await login(username.value, password.value);
  if (result.success) {
    if (isAdmin.value) {
      router.push("/main");
    } else {
      router.push("/dashboard");
    }
  } else {
    notice.value = result.message;
    isError.value = true;
    password.value = "";
  }
}
</script>

<template>
  <div class="login-wrapper">
    <!-- 배경 영상 -->
    <video class="bg-video" autoplay muted loop playsinline>
      <source src="/bg-emergency.mp4" type="video/mp4" />
    </video>

    <!-- 로그인 카드 -->
    <div class="login-card">
      <!-- 알림 배너 -->
      <div
        v-if="notice"
        :class="['notice-banner', isError ? 'notice-error' : 'notice-info']"
      >
        {{ notice }}
      </div>

      <!-- 로고 (경계 없이) -->
      <div class="card-logo">
        <img src="/logos/logo.png" alt="GoldenTime" class="logo-img" />
      </div>

      <!-- 사이트명 -->
      <p class="site-title">스마트 긴급차 통행방해차량 신고 사이트</p>

      <!-- 폼 -->
      <form class="login-form" @submit.prevent="handleLogin">
        <div class="form-group">
          <input
            v-model="username"
            type="text"
            placeholder="아이디 입력"
            autocomplete="username"
            required
          />
        </div>
        <div class="form-group">
          <input
            v-model="password"
            type="password"
            placeholder="비밀번호 입력"
            autocomplete="current-password"
            required
          />
        </div>
        <button type="submit" class="btn-login">로그인</button>
      </form>

      <button class="btn-register" @click="$router.push('/register')">
        회원가입
      </button>
    </div>
  </div>
</template>

<style scoped>
/* ===== 배경 ===== */
.login-wrapper {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #040e24;
  position: relative;
  overflow: hidden;
}

.bg-video {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 100%;
  height: 100%;
  object-fit: cover;
  transform: translate(-50%, -50%);
  opacity: 0.45;
  z-index: 0;
}

/* ===== 카드 ===== */
.login-card {
  position: relative;
  z-index: 1;
  width: 100%;
  max-width: 420px;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  padding: 50px 40px;
  border-radius: 4px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  box-shadow: 0 25px 50px rgba(0, 0, 0, 0.4);
  text-align: center;
}

.card-logo {
  margin-bottom: 20px;
}

.logo-img {
  height: 55px;
  width: auto;
  filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
}

.site-title {
  font-size: 1.1rem;
  font-weight: 700;
  color: #ffffff;
  margin-bottom: 35px;
  letter-spacing: -0.01em;
  text-shadow: 0 2px 4px rgba(0,0,0,0.5);
}

/* ===== 폼 ===== */
.login-form {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.form-group input {
  width: 100%;
  padding: 15px 18px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  background: rgba(255, 255, 255, 0.1);
  color: #ffffff;
  font-size: 1rem;
  outline: none;
  transition: all 0.2s;
  box-sizing: border-box;
  text-transform: none !important;
}

.form-group input::placeholder {
  color: rgba(255, 255, 255, 0.6);
}

.form-group input:focus {
  border-color: rgba(255, 255, 255, 0.5);
  background: rgba(255, 255, 255, 0.15);
  box-shadow: 0 0 15px rgba(255, 255, 255, 0.1);
}

.btn-login {
  width: 100%;
  padding: 15px;
  background: #1976d2;
  color: #fff;
  border: none;
  font-size: 1.05rem;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.2s;
  margin-top: 10px;
}

.btn-login:hover {
  background: #1e88e5;
  transform: translateY(-1px);
  box-shadow: 0 5px 15px rgba(25, 118, 210, 0.4);
}

.btn-register {
  margin-top: 28px;
  background: none;
  border: none;
  color: rgba(255, 255, 255, 0.7);
  font-size: 0.9rem;
  text-decoration: underline;
  cursor: pointer;
}

.btn-register:hover {
  color: #ffffff;
}

/* 알림 */
.notice-banner {
  padding: 12px;
  margin-bottom: 20px;
  font-size: 0.88rem;
  font-weight: 600;
}

.notice-info {
  background: #e0f2fe;
  color: #0369a1;
}

.notice-error {
  background: #fee2e2;
  color: #b91c1c;
}
</style>
