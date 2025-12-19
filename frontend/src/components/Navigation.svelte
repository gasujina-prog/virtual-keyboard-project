<script>
    import { link } from 'svelte-spa-router'
    import { is_login, username, access_token } from '../lib/store'
    import { push } from 'svelte-spa-router'

    // 로그아웃 함수
    const logout = () => {
        access_token.set('')
        username.set('')
        is_login.set(false)
        push('/')
    }
</script>

<nav class="navbar navbar-expand-lg navbar-light bg-light border-bottom">
    <div class="container-fluid">

        <a class="navbar-brand fw-bold" use:link href="/">
            Project Keyboard
        </a>

        <span class="fs-6 text-muted fw-normal d-none d-lg-inline-block ms-1 me-4"
              style="cursor: default; user-select: none;">
            | 웹캠 하나로 즐기는 가상 키보드
        </span>

        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
            <span class="navbar-toggler-icon"></span>
        </button>

        <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav me-auto mb-2 mb-lg-0">
                <li class="nav-item">
                    <a class="nav-link" use:link href="/board">자유게시판</a>
                </li>
                {#if $is_login}
                    <li class="nav-item">
                        <a class="nav-link" href="#" on:click={logout}>로그아웃 ({$username})</a>
                    </li>
                {:else}
                    <li class="nav-item">
                        <a class="nav-link" use:link href="/user-login">로그인</a>
                    </li>
                {/if}
            </ul>
        </div>
    </div>
</nav>