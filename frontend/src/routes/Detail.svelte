<script>
    import fastapi from "../lib/api"
    import { link, push } from 'svelte-spa-router'
    import { is_login, username } from '../lib/store'
    import moment from "moment/min/moment-with-locales"
    moment.locale('ko')
    
    export let params = {}
    let question_id = params.question_id

    // [수정 핵심] 초기값을 꼼꼼하게 채워둬야 에러가 안 납니다.
    let question = {
        id: 0,
        subject: "",
        content: "",
        create_date: "",
        answers: [],
        voter: [],
        user: { username: '' }
    }

    let content = ""
    let error = {detail:[]}

    function get_question() {
        fastapi("get", "/api/question/detail/" + question_id, {}, (json) => {
            question = json
        })
    }

    get_question()

    function post_answer(event) {
        event.preventDefault()
        let url = "/api/answer/create/" + question_id
        let params = {
            content: content,
            username: $username
        }
        fastapi('post', url, params,
            (json) => {
                content = ''
                error = {detail:[]}
                get_question()
                // 댓글 등록 시 스크롤 맨 아래로
                setTimeout(() => window.scrollTo(0, document.body.scrollHeight), 100)
            },
            (err_json) =>{
                error = err_json
                alert(err_json.detail)
            }
        )
    }

    function delete_question(_question_id) {
        if(window.confirm('정말로 삭제하시겠습니까?')) {
            let url = "/api/question/delete/" + _question_id
            let params = { username: $username }
            fastapi('delete', url, params,
                (json) => { push('/board') },
                (err_json) => { error = err_json; alert(err_json.detail) }
            )
        }
    }

    function delete_answer(answer_id) {
        if(window.confirm('정말로 삭제하시겠습니까?')) {
            let url = "/api/answer/delete"
            let params = { answer_id: answer_id }
            fastapi('delete', url, params,
                (json) => { get_question() },
                (err_json) => { error = err_json; alert(err_json.detail) }
            )
        }
    }

    function vote_question(_question_id) {
        if(window.confirm('정말로 추천하시겠습니까?')) {
            let url = "/api/question/vote"
            let params = { question_id: _question_id }
            fastapi('post', url, params,
                (json) => { get_question() },
                (err_json) => { error = err_json; alert(err_json.detail) }
            )
        }
    }

    function vote_answer(answer_id) {
        if(window.confirm('정말로 추천하시겠습니까?')) {
            let url = "/api/answer/vote"
            let params = { answer_id: answer_id }
            fastapi('post', url, params,
                (json) => { get_question() },
                (err_json) => { error = err_json; alert(err_json.detail) }
            )
        }
    }
</script>

<div class="chat-container">
    <div class="chat-header sticky-top bg-white border-bottom p-3 d-flex justify-content-between align-items-center">
        <div class="d-flex align-items-center gap-2">
            <button class="btn btn-sm btn-light rounded-circle" on:click="{() => push('/board')}">⬅️</button>
            <h5 class="m-0 fw-bold text-truncate" style="max-width: 250px;">{question.subject}</h5>
            <span class="badge bg-secondary rounded-pill">{question.answers.length}</span>
        </div>
        <button class="btn btn-sm btn-light" on:click="{get_question}">🔄</button>
    </div>

    <div class="chat-body p-3">

        <div class="chat-bubble-wrapper left">
            <div class="profile-icon">👑</div>
            <div class="bubble-content">
                <div class="sender-name">
                    {question.user ? question.user.username : "(알수없음)"}
                    <span class="date">{question.create_date ? moment(question.create_date).format("LT") : ""}</span>
                </div>
                <div class="bubble main-post">
                    {@html question.content ? question.content.replace(/\n/g, '<br>') : ""}
                </div>

                <div class="bubble-actions mt-1">
                    <button class="btn btn-sm btn-link text-decoration-none p-0 me-2" on:click="{() => vote_question(question.id)}">
                        ❤️ {question.voter.length}
                    </button>
                    {#if question.user && $username === question.user.username}
                        <a use:link href="#/question-modify/{question.id}" class="text-muted small me-1">수정</a>
                        <span class="text-muted small">|</span>
                        <button class="btn btn-link text-muted small p-0 ms-1 text-decoration-none" on:click="{() => delete_question(question.id)}">삭제</button>
                    {/if}
                </div>
            </div>
        </div>

        <div class="text-center my-4">
            <span class="badge bg-light text-secondary rounded-pill px-3">여기부터 댓글 대화</span>
        </div>

        {#each question.answers as answer}
            <div class="chat-bubble-wrapper {answer.user && answer.user.username === $username ? 'right' : 'left'}">
                {#if !(answer.user && answer.user.username === $username)}
                    <div class="profile-icon">👤</div>
                {/if}

                <div class="bubble-content">
                    {#if !(answer.user && answer.user.username === $username)}
                        <div class="sender-name">{answer.user ? answer.user.username : "익명"}</div>
                    {/if}

                    <div class="bubble {answer.user && answer.user.username === $username ? 'my-bubble' : 'other-bubble'}">
                        {@html answer.content.replace(/\n/g, '<br>')}
                    </div>

                    <div class="bubble-info d-flex align-items-center gap-2 mt-1 {answer.user && answer.user.username === $username ? 'justify-content-end' : ''}">
                        <span class="date">{moment(answer.create_date).fromNow()}</span>

                        <button class="btn btn-sm p-0 border-0" on:click="{() => vote_answer(answer.id)}">
                            👍 <small>{answer.voter.length}</small>
                        </button>

                        {#if answer.user && $username === answer.user.username}
                            <a use:link href="/answer-modify/{answer.id}" class="text-muted small">✏️</a>
                            <button class="btn btn-sm p-0 border-0" on:click="{() => delete_answer(answer.id)}">🗑️</button>
                        {/if}
                    </div>
                </div>
            </div>
        {/each}
    </div>

    <div class="chat-input-area bg-white border-top p-2">
        <form class="d-flex gap-2 align-items-end" on:submit="{post_answer}">
            <textarea
                class="form-control"
                rows="2"
                placeholder={$is_login ? "메시지를 입력하세요..." : "로그인이 필요합니다."}
                bind:value={content}
                disabled={!$is_login}
                style="resize: none; border-radius: 15px;"></textarea>
            <button class="btn btn-warning rounded-circle" style="width: 50px; height: 50px;" disabled={!$is_login}>
                ➤
            </button>
        </form>
    </div>
</div>

<style>
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        background-color: #1f1e33;
        min-height: 80vh;
        display: flex;
        flex-direction: column;
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(0,0,0,0.1);
        overflow: hidden;
    }

    .chat-body {
        flex: 1;
        overflow-y: auto;
    }

    .chat-bubble-wrapper {
        display: flex;
        margin-bottom: 15px;
        gap: 10px;
    }

    .profile-icon {
        width: 40px;
        height: 40px;
        background-color: #fff;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        border: 1px solid #ddd;
    }

    .bubble-content {
        max-width: 70%;
        display: flex;
        flex-direction: column;
    }

    .bubble {
        padding: 10px 15px;
        border-radius: 15px;
        font-size: 0.95rem;
        position: relative;
        word-break: break-word;
    }

    .chat-bubble-wrapper.left {
        align-items: flex-start;
    }
    .other-bubble, .main-post {
        background-color: #ffffff;
        border-top-left-radius: 2px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .sender-name {
        font-size: 0.8rem;
        color: #555;
        margin-bottom: 4px;
    }

    .chat-bubble-wrapper.right {
        justify-content: flex-end;
    }
    .my-bubble {
        background-color: #fef01b;
        border-top-right-radius: 2px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }

    .date {
        font-size: 0.7rem;
        color: #888;
        margin-left: 5px;
        margin-right: 5px;
    }

    .main-post {
        font-size: 1rem;
        border: 2px solid #fff;
    }

    .chat-input-area {
        position: sticky;
        bottom: 0;
    }
</style>