<script>
  import Router from 'svelte-spa-router'
  import Navigation from "./components/Navigation.svelte"

  // 라우터 설정
  import Home from "./routes/Home.svelte"
  import Board from "./routes/Board.svelte"
  import Detail from "./routes/Detail.svelte"
  import QuestionCreate from "./routes/QuestionCreate.svelte"
  import UserCreate from "./routes/UserCreate.svelte"
  import UserLogin from "./routes/UserLogin.svelte"
  import QuestionModify from "./routes/QuestionModify.svelte"
  import AnswerModify from "./routes/AnswerModify.svelte"
  import Game from "./routes/Game.svelte"
  import Mypage from "./routes/Mypage.svelte"

  const routes = {
    '/': Home,
    '/board': Board,
    '/detail/:question_id': Detail,
    '/question-create': QuestionCreate,
    '/user-create' : UserCreate,
    '/user-login' : UserLogin,
    '/question-modify/:question_id': QuestionModify,
    '/answer-modify/:answer_id' : AnswerModify,
    '/game': Game,
    '/mypage': Mypage
  }
</script>

<Navigation />

<Router {routes}/>

<style>
  /* 전역 스타일이 필요하다면 여기에 작성 */
</style>