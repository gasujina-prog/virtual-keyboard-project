<script>
    import { push } from 'svelte-spa-router'
    import { is_login } from "../lib/store"
    import { onMount, onDestroy, tick } from 'svelte'
    import { slide, fade } from 'svelte/transition' // fade 효과 추가
    import fastapi from "../lib/api"

    let testInput = ""
    let sseSource = null
    let show_camera = false // 기본 상태는 접힘
    let textareaBox;

    // 카메라 토글 함수
    const toggleCamera = () => {
        show_camera = !show_camera
        let url = "/api/keyboard/toggle"
        let params = { active: show_camera }
        fastapi('post', url, params, () => {}, () => {})
    }

    // SSE 연결 및 키 입력 처리 (변동 없음)
    onMount(() => {
        if ($is_login) {
            sseSource = new EventSource('http://127.0.0.1:5000/stream')

            // 1. 여기에 'async'를 꼭 붙여야 await를 쓸 수 있습니다.
            sseSource.onmessage = async (event) => {
                const key = event.data

                if (key === 'Backspace') {
                    testInput = testInput.slice(0, -1)
                } else if (key === 'SpaceBar') {
                    testInput += " "
                } else if (key === 'Enter') {
                    testInput += "\n"
                } else if (key.length === 1) {
                    testInput += key
                }

                // 2. 어떤 키를 눌렀든, 화면이 갱신된 후에는 항상 스크롤을 맨 밑으로 내립니다.
                await tick();
                if (textareaBox) {
                    textareaBox.scrollTop = textareaBox.scrollHeight;
                }
            }
        }
    })

    onDestroy(() => {
        if (sseSource) {
            sseSource.close()
        }
    })
</script>

<div class="container mt-4 mb-5">

    {#if $is_login}
        <div class="border rounded shadow-sm bg-white p-4">

            <div class="row g-4"> <div class="col-lg-6 order-lg-1 order-2">

                    <button class="btn {show_camera ? 'btn-danger' : 'btn-primary'} w-100 py-2 fw-bold shadow-sm mb-3"
                            on:click="{toggleCamera}">
                        {show_camera ? '⏹ 카메라 접기 (USB 입력 모드)' : '📸 카메라 펼치기 (가상 입력 모드)'}
                    </button>

                    {#if show_camera}
                        <div class="d-flex flex-column gap-3" transition:slide|local>
                            <div class="video-box p-2 bg-light border rounded text-center">
                                <h6 class="text-muted mb-2 small">👇 내 모습 (Webcam)</h6>
                                <div class="img-container">
                                    <img src="http://127.0.0.1:5000/video_feed_cam" class="img-fluid rounded" alt="카메라 화면" />
                                </div>
                            </div>
                            <div class="video-box p-2 bg-light border rounded text-center">
                                <h6 class="text-muted mb-2 small">👇 가상 키보드 인식 (Warp View)</h6>
                                <div class="img-container">
                                    <img src="http://127.0.0.1:5000/video_feed_warp" class="img-fluid rounded" alt="키보드 화면" />
                                </div>
                            </div>
                        </div>
                    {:else}
                        <div class="alert alert-light border text-center p-5 text-muted" transition:fade>
                            <h3 class="mb-3">📷</h3>
                            버튼을 눌러 카메라를 실행해주세요.
                        </div>
                    {/if}

                </div>

                <div class="col-lg-6 order-lg-2 order-1 d-flex flex-column">

                    <div class="flex-grow-1"> <h4 class="mb-3">⌨️ 가상 키보드 제어판</h4>
                        <p class="text-muted small mb-4">
                            왼쪽의 카메라를 켜고 가상 키보드를 쳐보세요.<br>
                            인식된 위치를 아래에 띄우며 오른쪽 입력창에 나타납니다.<br>
                            (USB 키보드로도 수정이 가능합니다)
                        </p>

                        <textarea class="form-control form-control-lg mb-4"
                                  rows="8"
                                  placeholder="여기에 타이핑 결과가 표시됩니다..."
                                  bind:value={testInput}
                                  bind:this={textareaBox}
                                  style="resize: none; font-family: monospace;"></textarea>
                    </div>

                    <div class="d-grid gap-2 d-md-flex justify-content-md-end mt-auto">
                        <button class="btn btn-success flex-grow-1" on:click={() => push('/game')}>
                            🎮 게임 시작
                        </button>
                        <button class="btn btn-success text-white flex-grow-1" on:click={() => push('/mypage')}>
                            👤 마이페이지
                        </button>
                    </div>

                </div>

            </div> </div>

    {:else}
        <div class="text-center py-5">
            <h1 class="display-4 fw-bold text-primary mb-3">Project Keyboard</h1>
            <p class="lead mb-5">
                웹캠 하나로 즐기는<br>
                가상 키보드 & 타자 연습 플랫폼
            </p>
            <div class="d-grid gap-2 d-sm-flex justify-content-sm-center">
                <button class="btn btn-primary btn-lg px-4 gap-3" on:click="{() => push('/user-login')}">
                    로그인 하고 시작하기
                </button>
                <button class="btn btn-outline-secondary btn-lg px-4" on:click="{() => push('/board')}">
                    게시판 구경하기
                </button>
            </div>
        </div>
    {/if}

</div>

<style>
    /* 이미지 비율 유지 및 잘림 방지 */
    .img-container {
        width: 100%;
        /* 필요에 따라 높이를 고정하거나 비율을 설정할 수 있습니다. 예: aspect-ratio: 4/3; */
        overflow: hidden;
        background: #000; /* 로딩 중이나 빈 공간 검은색 처리 */
        border-radius: 0.25rem;
    }
    img {
        width: 100%;
        height: auto;
        display: block;
        /* object-fit: contain; 이미지가 잘리지 않게 하려면 이 속성을 사용하세요 */
    }
</style>