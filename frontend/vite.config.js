import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

// https://vite.dev/config/
export default defineConfig({
  plugins: [svelte()],
  base: '/static/assets/',
  build: {


    // 1. 결과물이 나올 위치 (여기가 제일 중요!)
    outDir: '../static/assets',
    
    // 2. assets 폴더 안에 또 assets 폴더가 생기는 걸 방지 (지저분함 방지)
    assetsDir: '.', 
    
  }
})