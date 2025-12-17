<script>
    import { push } from 'svelte-spa-router'
    import fastapi from "../lib/api"
    import Error from "../components/Error.svelte"
    import { username } from "../lib/store"

    let error = {detail:[]}
    let subject = ''
    let content = ''

    function post_question(event) {
        event.preventDefault()
        let url = "/api/question/create"
        let params = {
            subject: subject,
            content: content,
            username: $username
        }
        fastapi('post', url, params,
            (json) => {
                push("/board") // [변경] 저장 후에도 게시판 목록으로 이동
            },
            (json_error) => {
                error = json_error
            }
        )
    }

    // [추가] 취소 버튼 기능
    function cancel() {
        push('/board') // 게시판 목록으로 즉시 이동
    }
</script>

<div class="container">
    <h5 class="my-3 border-bottom pb-2">질문 등록</h5>
    <Error error={error} />
    <form method="post" class="my-3">
        <div class="mb-3">
            <label for="subject">제목</label>
            <input type="text" class="form-control" id="subject" bind:value="{subject}">
        </div>
        <div class="mb-3">
            <label for="content">내용</label>
            <textarea class="form-control" id="content" rows="10" bind:value="{content}"></textarea>
        </div>

        <div class="d-flex gap-2">
            <button class="btn btn-primary" on:click="{post_question}">저장하기</button>

            <button type="button" class="btn btn-secondary" on:click="{cancel}">취소</button>
        </div>
    </form>
</div>