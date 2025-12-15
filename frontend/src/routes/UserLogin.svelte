<script>
  import { push } from 'svelte-spa-router'
  import { login } from '../lib/api'
  import { username, is_login, access_token } from '../lib/store'

  let login_username = ""
  let login_password = ""
  let error = {detail:[]}

  const get_login = () => {
    let params = {
      username: login_username,
      password: login_password,
    }
    login('api/v1/user/login', params).then(response => {
        $access_token = response.access_token
        $username = response.username
        $is_login = true
        push('/')
    }).catch(err => {
        error = err
    })
  }
</script>

<div class="container">
  <h5 class="my-3 border-bottom pb-2">로그인</h5>
  {#if error.detail.length > 0}
  <div class="alert alert-danger" role="alert">
      <div>{error.detail}</div>
  </div>
  {/if}

  <form on:submit|preventDefault={get_login}>
    <div class="mb-3">
      <label for="username">사용자ID</label>
      <input type="text" class="form-control" id="username" bind:value={login_username}>
    </div>

    <div class="mb-3">
      <label for="password">비밀번호</label>
      <input type="password" class="form-control" id="password" bind:value={login_password}>

      <div class="mt-2 text-end">
        <span class="text-muted small">계정이 없으신가요?</span>
        <a href="#/user-create" class="small ms-1 text-decoration-none">회원가입</a>
      </div>
    </div>

    <button type="submit" class="btn btn-primary w-100">로그인</button>
  </form>
</div>