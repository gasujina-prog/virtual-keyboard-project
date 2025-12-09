<script>
  import Router from 'svelte-spa-router'
  import Home from "./routes/Home.svelte"
  import Detail from "./routes/Detail.svelte"
  import QuestionCreate from "./routes/QuestionCreate.svelte"
  import Navigation from "./components/Navigation.svelte"
  import UserCreate from "./routes/UserCreate.svelte"
  import UserLogin from "./routes/UserLogin.svelte"
  import QuestionModify from "./routes/QuestionModify.svelte"
  import AnswerModify from "./routes/AnswerModify.svelte"


  const routes = {
    '/': Home,
    '/detail/:question_id': Detail,
    '/question-create': QuestionCreate,
    '/user-create' : UserCreate,
    '/user-login' : UserLogin,
    '/question-modify/:question_id': QuestionModify,
    '/answer-modify/:answer_id' : AnswerModify
  }
</script>

<Navigation />

<div class="camera-section">
  <h2>실시간 가상 키보드 시스템</h2>
  <div class="video-container">
    
    <div class="video-box">
      <h4>Camera View</h4>
      <img src="/video_feed_cam" alt="카메라 화면 연결 대기중..." />
    </div>

    <div class="video-box">
      <h4>Virtual Keyboard</h4>
      <img src="/video_feed_warp" alt="키보드 화면 연결 대기중..." />
    </div>

  </div>
  <hr /> </div>

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
    flex-wrap: wrap; /* 화면 좁으면 줄바꿈 */
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
    width: 480px; /* 원하는 크기로 조절 가능 */
    border-radius: 5px;
    display: block;
  }
</style>
