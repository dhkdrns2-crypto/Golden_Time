<script setup>
import { computed, onBeforeMount } from 'vue'
import { useRoute } from 'vue-router'
import { useAuth } from './store/auth'
import AppLayout from './components/AppLayout.vue'

const route = useRoute()
const { isLoggedIn, fetchMe } = useAuth()

onBeforeMount(async () => {
  await fetchMe()
})

const useLayout = computed(() => {
  return isLoggedIn.value && route.path !== '/login' && route.path !== '/register'
})
</script>

<template>
  <AppLayout v-if="useLayout">
    <router-view />
  </AppLayout>
  <router-view v-else />
</template>
