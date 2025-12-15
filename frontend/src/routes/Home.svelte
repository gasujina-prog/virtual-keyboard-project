<script>
    import { push } from 'svelte-spa-router'
    import { is_login } from "../lib/store"
    import { onMount, onDestroy } from 'svelte'

    // [추가] 테스트용 텍스트 변수
    let testInput = ""
    let sseSource = null

    // [추가] 화면이 켜지면 서버와 연결 (SSE)
    onMount(() => {
        if ($is_login) {
            // 서버의 스트림 주소로 연결
            sseSource = new EventSource('http://127.0.0.1:5000/stream')

            sseSource.onmessage = (event) => {
                const key = event.data
                console.log("Input received:", key)

                if (key === 'Backspace') {
                    testInput = testInput.slice(0, -1)
                } else if (key === 'SpaceBar') { // 스페이스바 처리
                    testInput += " "
                } else if (key === 'Enter') {
                    testInput += "\n"
                } else if (key.length === 1) { // 일반 문자만 (Shift 등 제외)
                    testInput += key
                }
            }
        }
    })

    // [추가] 화면 나가면 연결 끊기 (리소스 낭비 방지)
    onDestroy(() => {
        if (sseSource) {
            sseSource.close()
        }
    })

</script>

<div class="container text-center mt-5">
    <h1 class="display-4 text-primary fw-bold">Project Keyboard</h1>
    <p class="lead mb-4">
        웹캠 하나로 즐기는<br>
        가상 키보드 & 타자 연습 플랫폼
    </p>

    {#if $is_login}
        <div class="border rounded p-4 shadow-sm bg-white mx-auto mb-4" style="max-width: 600px;">
            <h4 class="mb-3">⌨️ 입력 테스트 존</h4>
            <p class="text-muted small">카메라를 켜고 가상 키보드를 눌러보세요!<br>(USB 키보드로도 입력/수정이 가능합니다)</p>

            <input type="text" class="form-control form-control-lg text-center mb-3"
                   placeholder="여기에 타이핑됩니다..."
                   bind:value={testInput}>

            <div class="d-grid gap-2">
                <button class="btn btn-success btn-lg" disabled>
                    🎮 게임 시작 (준비중)
                </button>
                <button class="btn btn-warning btn-lg text-white" on:click={() => alert('마이페이지 기능은 준비 중입니다!')}>
                    👤 마이페이지
                </button>
            </div>
        </div>

    {:else}
        <div class="d-grid gap-2 d-sm-flex justify-content-sm-center">
            <button class="btn btn-primary btn-lg px-4 gap-3" on:click="{() => push('/user-login')}">
                로그인 하고 시작하기
            </button>
            <button class="btn btn-outline-secondary btn-lg px-4" on:click="{() => push('/board')}">
                게시판 구경하기
            </button>
        </div>
    {/if}

</div>