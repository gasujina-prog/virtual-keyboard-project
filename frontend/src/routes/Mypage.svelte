<script>
    import { onMount } from 'svelte'
    import { push } from 'svelte-spa-router'
    import { is_login, username, access_token } from "../lib/store"
    import { fade } from 'svelte/transition'

    let stats = {
        total_games: 0,
        high_score: 0,
        avg_accuracy: 0,
        username: "User"
    }

    // [수정] 설정 변수 3개 통합
    let config = {
        sensitivity: 0.5,
        cooldown: 0.2,
        dwell: 0.1
    }

    let loading = true
    let isSaving = false

    // 모달 상태 변수들
    let showMyPostModal = false
    let showUpdateModal = false
    let showDeleteModal = false

    let myPosts = { questions: [], answers: [] }
    let modalTab = 'question'

    // 수정/탈퇴용 입력 변수
    let updateForm = { current_password: '', new_password: '', new_email: '' }
    let deletePassword = ''

    $: tier = getTier(stats.high_score)
    function getTier(score) {
        if (score >= 2000) return { name: 'DIAMOND', color: '#b9f2ff', icon: '💎' }
        if (score >= 1000) return { name: 'PLATINUM', color: '#e5e4e2', icon: '💿' }
        if (score >= 500)  return { name: 'GOLD', color: '#ffd700', icon: '🥇' }
        if (score >= 300)  return { name: 'SILVER', color: '#c0c0c0', icon: '🥈' }
        return { name: 'BRONZE', color: '#cd7f32', icon: '🥉' }
    }

    onMount(async () => {
        if (!$is_login) {
            alert("로그인이 필요합니다.")
            push('/user-login')
            return
        }
        await loadData()
    })

    async function loadData() {
        try {
            // 1. 통계 정보 불러오기
            const resStats = await fetch('/game/stats')
            if (resStats.ok) {
                const data = await resStats.json()
                if (data.result === 'success') stats = data
            }

            // 2. [수정] 설정값 3개 불러오기 (Config API 사용)
            const resConf = await fetch('/api/setting/config')
            if (resConf.ok) {
                const data = await resConf.json()
                if (data.result === 'success') {
                    config.sensitivity = data.sensitivity
                    config.cooldown = data.cooldown
                    config.dwell = data.dwell
                }
            }
        } catch (e) {
            console.error(e)
        } finally {
            loading = false
        }
    }

    // [수정] 설정값 3개 저장 함수 (Config API 사용)
    async function updateConfig() {
        isSaving = true
        try {
            await fetch('/api/setting/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            })
        } catch (e) {
            console.error(e)
        } finally {
            setTimeout(() => isSaving = false, 500)
        }
    }

    // 1. 내 글 불러오기
    async function openMyPostModal() {
        const res = await fetch('/api/myposts', { credentials: 'include' })
        if (res.ok) {
            myPosts = await res.json()
            showMyPostModal = true
        } else alert("불러오기 실패")
    }

    // 2. 회원 정보 수정 요청
    async function requestUpdate() {
        if (!updateForm.current_password) return alert("현재 비밀번호를 입력해주세요.")

        const res = await fetch('/api/user/update', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify(updateForm)
        })
        const json = await res.json()

        if (res.ok) {
            alert(json.message)
            showUpdateModal = false
            updateForm = { current_password: '', new_password: '', new_email: '' } // 초기화
        } else {
            alert(json.detail)
        }
    }

    // 3. 회원 탈퇴 요청
    async function requestDelete() {
        if (!deletePassword) return alert("비밀번호를 입력해주세요.")
        if (!confirm("정말로 탈퇴하시겠습니까? 모든 데이터가 삭제됩니다.")) return

        const res = await fetch('/api/user/delete', {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ password: deletePassword })
        })
        const json = await res.json()

        if (res.ok) {
            alert(json.message)
            // 로그아웃 처리
            is_login.set(false)
            access_token.set('')
            username.set('')
            push('/') // 홈으로 이동
        } else {
            alert(json.detail)
        }
    }

    const goFeature = (feature) => {
        if (feature === '내가 쓴 글 관리') openMyPostModal()
        else if (feature === '회원 정보 수정') showUpdateModal = true
        else if (feature === '회원 탈퇴') showDeleteModal = true
    }
</script>

<div class="container mt-5" style="max-width: 700px;">
    {#if loading}
        <div class="text-center text-white mt-5"><div class="spinner-border text-warning"></div></div>
    {:else}
        <div class="d-flex flex-column gap-4" transition:fade>

            <div class="card bg-dark text-white border-secondary shadow">
                <div class="card-body p-4 d-flex align-items-center justify-content-between">
                    <div class="d-flex align-items-center gap-3">
                        <div class="display-3">{tier.icon}</div>
                        <div>
                            <h3 class="fw-bold mb-0" style="color: {tier.color}">{tier.name}</h3>
                            <div class="text-muted small">ID: {stats.username}</div>
                        </div>
                    </div>
                    <div class="text-end">
                        <div class="text-warning small fw-bold">HIGH SCORE</div>
                        <div class="display-4 fw-bold">{stats.high_score}</div>
                    </div>
                </div>
                <div class="card-footer bg-secondary bg-opacity-25 border-secondary d-flex justify-content-around py-3">
                    <div class="text-center">
                        <div class="text-white-50 small">총 플레이</div>
                        <div class="fw-bold">{stats.total_games} 판</div>
                    </div>
                    <div class="text-center">
                        <div class="text-white-50 small">평균 정확도</div>
                        <div class="fw-bold {stats.avg_accuracy >= 90 ? 'text-success' : 'text-warning'}">
                            {stats.avg_accuracy}%
                        </div>
                    </div>
                </div>
            </div>

            <div class="list-group shadow">
                <button class="list-group-item list-group-item-action list-group-item-dark p-3 d-flex justify-content-between align-items-center"
                        on:click={() => goFeature('회원 정보 수정')}>
                    <span>👤 회원 정보 수정</span>
                    <span class="text-muted">❯</span>
                </button>
                <button class="list-group-item list-group-item-action list-group-item-dark p-3 d-flex justify-content-between align-items-center"
                        on:click={() => goFeature('내가 쓴 글 관리')}>
                    <span>📝 내가 쓴 글 관리</span>
                    <span class="text-muted">❯</span>
                </button>
                <button class="list-group-item list-group-item-action list-group-item-danger p-3 d-flex justify-content-between align-items-center"
                        on:click={() => goFeature('회원 탈퇴')}>
                    <span class="fw-bold">❌ 회원 탈퇴</span>
                    <span class="text-danger-emphasis">❯</span>
                </button>
            </div>
        </div>
    {/if}

    {#if showMyPostModal}
        <div class="modal d-block" style="background: rgba(0,0,0,0.8);">
            <div class="modal-dialog modal-dialog-centered modal-lg">
                <div class="modal-content bg-dark text-white border-secondary">
                    <div class="modal-header border-secondary">
                        <h5 class="modal-title">📝 내가 쓴 글</h5>
                        <button type="button" class="btn-close btn-close-white" on:click={() => showMyPostModal=false}></button>
                    </div>
                    <div class="modal-body">
                        <ul class="nav nav-pills nav-fill mb-3">
                            <li class="nav-item"><a class="nav-link {modalTab==='question'?'active bg-warning text-dark':''}" href={'#'} on:click|preventDefault={()=>modalTab='question'}>질문</a></li>
                            <li class="nav-item"><a class="nav-link {modalTab==='answer'?'active bg-warning text-dark':''}" href={'#'} on:click|preventDefault={()=>modalTab='answer'}>답변</a></li>
                        </ul>
                        <div class="list-group list-group-flush" style="max-height: 300px; overflow-y: auto;">
                            {#if modalTab === 'question'}
                                {#each myPosts.questions as q}
                                    <button class="list-group-item list-group-item-action list-group-item-dark bg-transparent text-white" on:click={() => push(`/detail/${q.id}`)}>
                                        {q.subject} <small class="text-muted ms-2">{q.create_date}</small>
                                    </button>
                                {/each}
                            {:else}
                                {#each myPosts.answers as a}
                                    <button class="list-group-item list-group-item-action list-group-item-dark bg-transparent text-white" on:click={() => push(`/detail/${a.question_id}`)}>
                                        <small class="text-warning">[Re] {a.question_subject}</small><br>{a.content}
                                    </button>
                                {/each}
                            {/if}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    {/if}

    {#if showUpdateModal}
        <div class="modal d-block" style="background: rgba(0,0,0,0.8);">
            <div class="modal-dialog modal-dialog-centered">
                <div class="modal-content bg-dark text-white border-secondary">
                    <div class="modal-header border-secondary">
                        <h5 class="modal-title">👤 회원 정보 수정</h5>
                        <button class="btn-close btn-close-white" on:click={() => showUpdateModal=false}></button>
                    </div>
                    <div class="modal-body">
                        <div class="mb-3">
                            <label class="form-label text-warning">현재 비밀번호 (필수)</label>
                            <input type="password" class="form-control" bind:value={updateForm.current_password} placeholder="본인 확인용">
                        </div>
                        <hr class="border-secondary">
                        <div class="mb-3">
                            <label class="form-label">새 이메일</label>
                            <input type="email" class="form-control" bind:value={updateForm.new_email} placeholder="변경할 이메일 (선택)">
                        </div>
                        <div class="mb-3">
                            <label class="form-label">새 비밀번호</label>
                            <input type="password" class="form-control" bind:value={updateForm.new_password} placeholder="변경할 비밀번호 (선택)">
                        </div>
                    </div>
                    <div class="modal-footer border-secondary">
                        <button class="btn btn-secondary" on:click={() => showUpdateModal=false}>취소</button>
                        <button class="btn btn-warning" on:click={requestUpdate}>수정 완료</button>
                    </div>
                </div>
            </div>
        </div>
    {/if}

    {#if showDeleteModal}
        <div class="modal d-block" style="background: rgba(0,0,0,0.8);">
            <div class="modal-dialog modal-dialog-centered">
                <div class="modal-content bg-dark text-white border-danger">
                    <div class="modal-header border-danger">
                        <h5 class="modal-title fw-bold text-danger">❌ 회원 탈퇴</h5>
                        <button class="btn-close btn-close-white" on:click={() => showDeleteModal=false}></button>
                    </div>
                    <div class="modal-body">
                        <p class="text-danger fw-bold">탈퇴 시 모든 게임 기록과 작성한 글이 삭제됩니다.</p>
                        <p>정말 탈퇴하시려면 비밀번호를 입력해주세요.</p>
                        <input type="password" class="form-control" bind:value={deletePassword} placeholder="비밀번호 입력">
                    </div>
                    <div class="modal-footer border-danger">
                        <button class="btn btn-secondary" on:click={() => showDeleteModal=false}>취소</button>
                        <button class="btn btn-danger" on:click={requestDelete}>탈퇴하기</button>
                    </div>
                </div>
            </div>
        </div>
    {/if}
</div>