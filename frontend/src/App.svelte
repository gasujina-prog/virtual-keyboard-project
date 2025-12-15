<script>
  import Router from 'svelte-spa-router'
  import { is_login } from "./lib/store"
  import { slide } from 'svelte/transition' // [추가] 부드러운 슬라이드 효과

  import Home from "./routes/Home.svelte"
  import Board from "./routes/Board.svelte"
  import Detail from "./routes/Detail.svelte"
  import fastapi from "./lib/api" // [필수] API 호출용
  import QuestionCreate from "./routes/QuestionCreate.svelte"
  import Navigation from "./components/Navigation.svelte"
  import UserCreate from "./routes/UserCreate.svelte"
  import UserLogin from "./routes/UserLogin.svelte"
  import QuestionModify from "./routes/QuestionModify.svelte"
  import AnswerModify from "./routes/AnswerModify.svelte"

  const routes = {
    '/': Home,
    '/board': Board,
    '/detail/:question_id': Detail,
    '/question-create': QuestionCreate,
    '/user-create' : UserCreate,
    '/user-login' : UserLogin,
    '/question-modify/:question_id': QuestionModify,
    '/answer-modify/:answer_id' : AnswerModify
  }

  // [추가] 카메라 상태 변수 (기본값: 켜짐)
  let show_camera = true

  // [추가] 카메라 토글 및 서버 전송 함수
  const toggleCamera = () => {
    show_camera = !show_camera

    // 서버에 "지금 가상 키보드 쓸 거야/말 거야" 알려주기
    let url = "/api/keyboard/toggle"
    let params = {
        active: show_camera
    }
    // 성공/실패 콜백은 비워둠 (로그만 확인)
    fastapi('post', url, params, () => {}, () => {})
  }
</script>

<Navigation />

{#if $is_login}
<div class="camera-section">
  <div class="d-flex justify-content-center align-items-center gap-3 mb-3">
    <h2 class="m-0">실시간 가상 키보드 시스템</h2>

    <button class="btn btn-outline-primary btn-sm rounded-pill px-3"
            on:click="{toggleCamera}">
      {show_camera ? '🔼 카메라 접기 (USB 입력 모드)' : '🔽 카메라 펼치기 (가상 입력 모드)'}
    </button>
  </div>

  {#if show_camera}
  <div class="video-container" transition:slide>
    <div class="video-box">
      <h4>Camera View</h4>
      <img src="http://127.0.0.1:5000/video_feed_cam" alt="카메라 화면 연결 대기중..." />
    </div>

    <div class="video-box">
      <h4>Virtual Keyboard</h4>
      <img src="http://127.0.0.1:5000/video_feed_warp" alt="키보드 화면 연결 대기중..." />
    </div>
  </div>
  <hr />
  {/if}
</div>
{/if}

<Router {routes}/>

<style>
  .camera-section {
    text-align: center;
    margin-top: 20px;
    margin-bottom: 20px;
    padding: 0 10px;
  }

  .video-container {
    display: flex;
    justify-content: center;
    gap: 20px;
    flex-wrap: wrap;
    margin-top: 10px;
    margin-bottom: 20px;
  }

  .video-box {
    border: 1px solid #ccc;
    padding: 10px;
    border-radius: 10px;
    background-color: #f8f9fa;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
  }

  h4 {
    margin-top: 0;
    margin-bottom: 10px;
    color: #333;
  }

  img {
    max-width: 100%;
    height: auto;
    width: 480px;
    border-radius: 5px;
    display: block;
  }
</style>