function getCookie(name) {
  try {
    const match = document.cookie.match(new RegExp('(^|;\\s*)' + name + '=([^;]*)'))
    return match ? decodeURIComponent(match[2]) : ''
  } catch {
    return ''
  }
}

/** Spring CSRF 쿠키 값 (multipart 등에서 _csrf 필드로도 사용) */
export function getXsrfToken() {
  return getCookie('XSRF-TOKEN')
}

/**
 * Cookie 기반 세션 + CSRF(XSRF-TOKEN 쿠키) 조합을 위한 fetch 래퍼
 * - credentials: include 기본 적용
 * - GET 이외의 요청에 X-XSRF-TOKEN 헤더 자동 추가
 */
export async function csrfFetch(input, init = {}) {
  const method = (init.method || 'GET').toUpperCase()

  const headers = new Headers(init.headers || {})
  const isSafeMethod = method === 'GET' || method === 'HEAD' || method === 'OPTIONS'

  if (!isSafeMethod) {
    let token = getCookie('XSRF-TOKEN')
    // 토큰이 없으면 API GET 으로 세션에 CSRF 발급 (Vite 프록시 시에도 /api 는 백엔드로 감)
    if (!token) {
      await fetch('/api/users/me', { method: 'GET', credentials: 'include' }).catch(() => {})
      token = getCookie('XSRF-TOKEN')
    }
    if (!token) {
      await fetch('/api/notices', { method: 'GET', credentials: 'include' }).catch(() => {})
      token = getCookie('XSRF-TOKEN')
    }
    if (!token) {
      await fetch('/', { method: 'GET', credentials: 'include' }).catch(() => {})
      token = getCookie('XSRF-TOKEN')
    }
    if (token) headers.set('X-XSRF-TOKEN', token)
  }

  return fetch(input, {
    credentials: 'include',
    ...init,
    headers,
  })
}

