<script>
    import { onMount, onDestroy } from 'svelte'
    import { is_login } from "../lib/store"
    import { push } from 'svelte-spa-router'
    import { fade, fly } from 'svelte/transition'

    // ================= 상태 관리 변수들 =================
    let gameState = 'menu' // 'menu', 'playing', 'result'
    let gameMode = ''      // 'easy', 'normal', 'hard'
    let isChallenge = false

    // 게임 데이터
    let score = 0
    let lives = 5
    let maxLives = 5
    let targets = []
    let currentInput = ""

    // [추가] 시간 제한 (Easy 모드용)
    let timeLeft = 0
    let timerLoop = null

    // [추가] 실시간 입력 피드백 (Hard 모드 인식 확인용)
    let lastKeyDetected = ""
    let feedbackColor = ""

    // Normal 모드 전용 변수
    let wordQueue = []
    let currentWord = ""
    let gameHistory = []

    // 타이머/루프
    let gameLoop = null
    let spawnLoop = null
    let sseSource = null
    let nextId = 0

    // 단어장
    const WORD_DICT = ["PYTHON", "SVELTE", "FLASK", "KEYBOARD", "CAMERA", "OPENCV", "CODING", "PROJECT", "ALGORITHM", "DEBUG", "LINUX", "DOCKER", "SERVER", "CLIENT", "ROBOT", "VISION", "MATRIX", "TENSOR", "YOLO", "MODEL"]

    // ================= 게임 시작 =================
    const initGame = (mode) => {
        gameMode = mode
        score = 0
        currentInput = ""
        targets = []
        nextId = 0
        gameHistory = []
        lastKeyDetected = "" // 초기화

        maxLives = isChallenge ? 3 : 5
        lives = maxLives

        // 루프 초기화
        clearInterval(gameLoop)
        clearInterval(spawnLoop)
        clearInterval(timerLoop)

        if (mode === 'normal') {
            let shuffled = [...WORD_DICT].sort(() => 0.5 - Math.random())
            wordQueue = shuffled.slice(0, 5)
            nextNormalWord()
        } else if (mode === 'easy') {
            // [설정] Easy 모드: 3분(180초) 타임 어택
            timeLeft = 180
            timerLoop = setInterval(() => {
                timeLeft -= 1
                if (timeLeft <= 0) endGame(true) // 시간 종료 시 클리어 처리
            }, 1000)
        }

        gameState = 'playing'

        if (mode !== 'normal') {
            let speed = isChallenge ? 20 : 30
            let spawnRate = isChallenge ? 800 : (mode === 'easy' ? 1200 : 1800)
            gameLoop = setInterval(updatePhysics, speed)
            spawnLoop = setInterval(spawnObject, spawnRate)
        }
    }

    // ================= 물리기 엔진 =================
    const updatePhysics = () => {
        let baseSpeed = gameMode === 'easy' ? 0.2 : 0.25
        let dropSpeed = baseSpeed * (isChallenge ? 1.5 : 1.0)

        targets = targets.map(t => ({ ...t, y: t.y + dropSpeed }))

        const missed = targets.filter(t => t.y > 90)
        if (missed.length > 0) {
            // Easy 모드는 시간 제한이므로 생명이 깎이지 않음 (선택 사항, 여기선 깎이도록 유지하되 시간 위주)
            lives -= missed.length
            targets = targets.filter(t => t.y <= 90)
            if (lives <= 0) endGame()
        }
    }

    // ================= 오브젝트 생성 =================
    const spawnObject = () => {
        const x = Math.random() * 80 + 10
        const colors = ['#FF5733', '#33FF57', '#3357FF', '#F333FF', '#FFD700']
        const color = colors[Math.floor(Math.random() * colors.length)]

        if (gameMode === 'easy') {
            const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            const char = chars[Math.floor(Math.random() * chars.length)]
            targets = [...targets, { id: nextId++, text: char, x, y: -10, color }]
        } else if (gameMode === 'hard') {
            const text = WORD_DICT[Math.floor(Math.random() * WORD_DICT.length)]
            targets = [...targets, { id: nextId++, text, x, y: -10, color }]
        }
    }

    // ================= Normal 모드 로직 =================
    const nextNormalWord = () => {
        if (wordQueue.length === 0) {
            endGame(true)
            return
        }
        currentWord = wordQueue.pop()
        currentInput = ""
    }

    // ================= 입력 처리 =================
    onMount(() => {
        if (!$is_login) {
            alert("로그인이 필요합니다.")
            push('/')
            return
        }
        sseSource = new EventSource('http://127.0.0.1:5000/stream')
        sseSource.onmessage = (event) => {
            if (gameState !== 'playing') return

            const key = event.data

            // [추가] 실시간 입력 피드백 (모든 모드 공통)
            // 백스페이스나 엔터가 아닌 일반 문자일 때만 표시
            if (key.length === 1) {
                lastKeyDetected = key.toUpperCase()
                // 잠시 후 사라지게 효과
                clearTimeout(window.feedbackTimer)
                window.feedbackTimer = setTimeout(() => { lastKeyDetected = "" }, 500)
            }

            if (gameMode === 'easy') {
                const charKey = key.toUpperCase()
                if (charKey.length === 1 && charKey >= 'A' && charKey <= 'Z') {
                    hitTargetEasy(charKey)
                }
            } else if (gameMode === 'hard') {
                if (key === 'Backspace') {
                    currentInput = currentInput.slice(0, -1)
                } else if (key.length === 1) {
                    currentInput += key.toUpperCase()
                    checkHardMatch()
                }
            } else {
                if (key === 'Enter') {
                    submitNormalInput()
                } else if (key === 'Backspace') {
                    currentInput = currentInput.slice(0, -1)
                } else if (key.length === 1) {
                    currentInput += key.toUpperCase()
                }
            }
        }
    })

    const hitTargetEasy = (key) => {
        const idx = targets.sort((a, b) => b.y - a.y).findIndex(t => t.text === key)
        if (idx !== -1) {
            score += 10
            targets = targets.filter(t => t.id !== targets[idx].id)
            feedbackColor = "text-success" // 맞춤 표시
        } else {
            feedbackColor = "text-danger" // 틀림 표시
        }
    }

    const checkHardMatch = () => {
        const idx = targets.sort((a, b) => b.y - a.y).findIndex(t => t.text === currentInput)
        if (idx !== -1) {
            score += 30
            targets = targets.filter(t => t.id !== targets[idx].id)
            currentInput = ""
        }
    }

    const submitNormalInput = () => {
        const targetLen = currentWord.length
        const inputLen = currentInput.length

        let matchCount = 0
        const minLen = Math.min(targetLen, inputLen)
        for (let i = 0; i < minLen; i++) {
            if (currentWord[i] === currentInput[i]) matchCount++
        }

        const accuracy = Math.round((matchCount / targetLen) * 100)
        const isCorrect = accuracy === 100

        gameHistory.push({
            target: currentWord,
            input: currentInput,
            correct: isCorrect,
            acc: accuracy
        })

        if (isCorrect) {
            score += 50
        } else {
            lives -= 1
        }

        if (lives <= 0) {
            endGame(false)
        } else {
            nextNormalWord()
        }
    }

    const endGame = (isClear = false) => {
        gameState = 'result'
        clearInterval(gameLoop)
        clearInterval(spawnLoop)
        clearInterval(timerLoop) // 타이머 정지
        if (isClear) {
            score += lives * 100
        }
    }

    const goMenu = () => {
        gameState = 'menu'
        gameMode = ''
    }

    onDestroy(() => {
        if (sseSource) sseSource.close()
        clearInterval(gameLoop)
        clearInterval(spawnLoop)
        clearInterval(timerLoop)
    })

    // 분:초 변환 함수
    const formatTime = (seconds) => {
        const m = Math.floor(seconds / 60)
        const s = seconds % 60
        return `${m}:${s < 10 ? '0' : ''}${s}`
    }
</script>

<div class="container-fluid mt-4 px-4">
    <div class="row g-4">

        <div class="col-lg-5 order-lg-1 order-2">
            <div class="card shadow-sm h-100 border-0">
                <div class="card-body bg-dark rounded p-3 d-flex flex-column gap-3">
                    <div class="video-box border border-secondary rounded overflow-hidden position-relative">
                        <span class="badge bg-secondary position-absolute m-2">Webcam</span>
                        <img src="http://127.0.0.1:5000/video_feed_cam" class="img-fluid" alt="Webcam" />
                    </div>
                    <div class="video-box border border-success rounded overflow-hidden position-relative">
                        <span class="badge bg-success position-absolute m-2">Warp View</span>
                        <img src="http://127.0.0.1:5000/video_feed_warp" class="img-fluid" alt="Warp View" />
                    </div>
                </div>
            </div>
        </div>

        <div class="col-lg-7 order-lg-2 order-1">
            <div class="game-area border rounded bg-dark position-relative overflow-hidden shadow-lg mx-auto"
                 style="height: 650px; background: radial-gradient(circle at center, #1a2a6c, #b21f1f, #fdbb2d);">

                {#if gameState === 'menu'}
                    <div class="absolute-center text-center w-100" transition:fade>
                        <h1 class="display-3 text-white fw-bold mb-4 text-shadow">KEYBOARD WAR</h1>
                        <div class="d-grid gap-3 col-8 mx-auto mb-4">
                            <button class="btn btn-success btn-lg py-3 fw-bold" on:click={() => initGame('easy')}>
                                🌱 EASY : 알파벳 요격 (3분 타임어택)
                            </button>
                            <button class="btn btn-primary btn-lg py-3 fw-bold" on:click={() => initGame('normal')}>
                                🌊 NORMAL : 단어 격파 (타자 검정)
                            </button>
                            <button class="btn btn-danger btn-lg py-3 fw-bold" on:click={() => initGame('hard')}>
                                🔥 HARD : 산성비 (종합 시험!)
                            </button>
                        </div>
                        <div class="form-check form-switch d-inline-block p-3 bg-black bg-opacity-50 rounded">
                            <input class="form-check-input" type="checkbox" id="challengeMode" bind:checked={isChallenge}>
                            <label class="form-check-label text-warning fw-bold" for="challengeMode">
                                💀 챌린지 모드 (HP 3 / 속도 증가)
                            </label>
                        </div>
                    </div>

                {:else if gameState === 'playing'}
                    <div class="d-flex justify-content-between p-3 position-absolute w-100 top-0 start-0 z-2">
                        <div class="h4 text-warning fw-bold text-shadow">SCORE: {score}</div>
                        <div class="d-flex gap-4">
                            {#if gameMode === 'easy'}
                                <div class="h4 text-info fw-bold text-shadow">
                                    ⏳ {formatTime(timeLeft)}
                                </div>
                            {/if}
                            <div class="h4 text-danger fw-bold text-shadow">
                                {'❤️'.repeat(lives)}
                            </div>
                        </div>
                    </div>

                    {#if lastKeyDetected}
                        <div class="position-absolute start-50 translate-middle-x z-3 p-2" style="top: 15%;">
                            <div class="display-1 fw-bold {feedbackColor || 'text-white'} text-shadow opacity-75" transition:fade>
                                {lastKeyDetected}
                            </div>
                            <div class="text-white small text-center bg-dark px-2 rounded bg-opacity-50">Detected</div>
                        </div>
                    {/if}

                    {#if gameMode === 'normal'}
                        <div class="absolute-center text-center w-100">
                            <div class="text-white small mb-2 opacity-75">단어 입력 후 Enter (기회는 한 번!)</div>
                            <h1 class="display-1 fw-bold text-white mb-4 text-shadow">{currentWord}</h1>
                            <div class="display-4 text-warning fw-bold border-bottom border-2 border-warning d-inline-block px-4 pb-2" style="min-width: 300px; min-height: 80px;">
                                {currentInput}<span class="blink">|</span>
                            </div>
                        </div>
                    {:else}
                        {#each targets as target (target.id)}
                            <div class="target-item position-absolute fw-bold d-flex justify-content-center align-items-center shadow"
                                 style="left: {target.x}%; top: {target.y}%; background-color: {target.color}; transform: translate(-50%, -50%);"
                                 transition:fly>
                                {target.text}
                            </div>
                        {/each}
                        {#if gameMode === 'hard'}
                            <div class="position-absolute bottom-0 w-100 p-4 text-center bg-black bg-opacity-50">
                                <span class="h3 text-white me-3">INPUT:</span>
                                <span class="h3 text-warning fw-bold border-bottom px-3">{currentInput}</span>
                            </div>
                        {/if}
                        <div class="position-absolute bottom-0 start-0 w-100 bg-danger opacity-25" style="height: 10%;"></div>
                    {/if}

                {:else if gameState === 'result'}
                    <div class="absolute-center text-center w-100 bg-dark bg-opacity-95 h-100 d-flex flex-column justify-content-center overflow-auto p-5" transition:fade>
                        <h1 class="display-3 {lives > 0 ? 'text-success' : 'text-danger'} fw-bold mb-2">
                            {lives > 0 || (gameMode === 'easy' && timeLeft <= 0) ? 'Finish!' : 'Game Over'}
                        </h1>
                        <h4 class="text-white mb-4">Total Score: <span class="text-warning">{score}</span></h4>

                        {#if gameMode === 'normal' && gameHistory.length > 0}
                            <div class="table-responsive w-75 mx-auto mb-4 border rounded">
                                <table class="table table-dark table-hover mb-0 text-center">
                                    <thead class="table-light text-dark">
                                        <tr>
                                            <th>제시어</th>
                                            <th>내가 쓴 단어</th>
                                            <th>정확도</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {#each gameHistory as record}
                                            <tr>
                                                <td class="fw-bold">{record.target}</td>
                                                <td class="{record.correct ? 'text-success' : 'text-danger'}">
                                                    {record.input || '(입력없음)'}
                                                </td>
                                                <td>
                                                    {#if record.correct}
                                                        <span class="badge bg-success">O ({record.acc}%)</span>
                                                    {:else}
                                                        <span class="badge {record.acc > 0 ? 'bg-warning text-dark' : 'bg-danger'}">
                                                            {record.acc > 0 ? '△' : 'X'} ({record.acc}%)
                                                        </span>
                                                    {/if}
                                                </td>
                                            </tr>
                                        {/each}
                                    </tbody>
                                </table>
                            </div>
                        {/if}

                        <div>
                            <button class="btn btn-primary btn-lg px-5 py-3 rounded-pill me-2" on:click={() => initGame(gameMode)}>
                                다시 하기
                            </button>
                            <button class="btn btn-outline-light btn-lg px-5 py-3 rounded-pill" on:click={goMenu}>
                                메뉴로
                            </button>
                        </div>
                    </div>
                {/if}

            </div>
        </div>
    </div>
</div>

<style>
    .absolute-center { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); }
    .text-shadow { text-shadow: 2px 2px 5px rgba(0,0,0,0.8); }
    .target-item { padding: 10px 20px; border-radius: 30px; color: white; font-size: 1.5rem; border: 2px solid white; box-shadow: 0 0 15px rgba(255, 255, 255, 0.6); white-space: nowrap; }
    .blink { animation: blinker 1s linear infinite; }
    @keyframes blinker { 50% { opacity: 0; } }
    .game-area { background-size: 400% 400%; animation: gradientBG 15s ease infinite; }
    @keyframes gradientBG { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
    .video-box img { width: 100%; display: block; }
    .z-2 { z-index: 2; }
    .z-3 { z-index: 3; }
</style>