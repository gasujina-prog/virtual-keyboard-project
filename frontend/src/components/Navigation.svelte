<script>
    import { link } from 'svelte-spa-router'
    import { is_login, username } from '../lib/store' // 스토어 경로 확인 필요 (lib/store 또는 ../store)
    import { push } from 'svelte-spa-router'
    import { access_token } from '../lib/store'

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
        <a class="navbar-brand" use:link href="/">Project Keyboard</a>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
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