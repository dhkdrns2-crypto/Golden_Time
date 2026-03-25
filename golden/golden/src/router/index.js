import { createRouter, createWebHashHistory } from 'vue-router'
import { useAuth } from '../store/auth'

import LoginView from '../views/LoginView.vue'
import RegisterView from '../views/RegisterView.vue'
import DashboardView from '../views/DashboardView.vue'
import MyPageView from '../views/MyPageView.vue'
import InquiryView from '../views/InquiryView.vue'
import NoticeView from '../views/NoticeView.vue'
import UserManagementView from '../views/UserManagementView.vue'
import MainView from '../views/MainView.vue'

const routes = [
  { path: '/', redirect: '/login' },
  { path: '/main', component: MainView, meta: { requiresAuth: true } },
  { path: '/login', component: LoginView, meta: { guest: true } },
  { path: '/register', component: RegisterView, meta: { guest: true } },
  { path: '/dashboard', component: DashboardView, meta: { requiresAuth: true } },
  { path: '/users', component: UserManagementView, meta: { requiresAuth: true, adminOnly: true } },
  { path: '/mypage', component: MyPageView, meta: { requiresAuth: true } },
  { path: '/inquiry', component: InquiryView, meta: { requiresAuth: true } },
  { path: '/notice', component: NoticeView, meta: { requiresAuth: true } },
]

const router = createRouter({
  history: createWebHashHistory(),
  routes,
})

router.beforeEach(async (to) => {
  const { isLoggedIn, isAdmin, isInitialized, fetchMe } = useAuth()

  // 초기 로딩 시 인증 정보 확인 대기
  if (!isInitialized.value) {
    await fetchMe()
  }

  if (to.meta.requiresAuth && !isLoggedIn.value) {
    return '/login'
  }

  // 일반 사용자가 메인화면(대시보드) 접근 시 신고목록으로 리다이렉트
  if (to.path === '/main' && isLoggedIn.value && !isAdmin.value) {
    return '/dashboard'
  }

  if (to.meta.adminOnly && !isAdmin.value) {
    return '/dashboard'
  }
  if (to.meta.guest && isLoggedIn.value) {
    if (isAdmin.value) return '/main'
    return '/dashboard'
  }
})

export default router
