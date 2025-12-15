<script>
    import fastapi from "../lib/api"
    // [수정] push 추가
    import { link, push } from 'svelte-spa-router'
    import { page, is_login } from "../lib/store"
    import moment from "moment/min/moment-with-locales"
    moment.locale('ko')

    let question_list = []
    let size = 10
    let total = 0
    $: total_page = Math.ceil(total/size)

    function get_question_list(_page) {
        let params = {
            page: _page,
            size: size,
        }
        fastapi('get', '/api/question/list', params, (json) => {
            question_list = json.question_list
            $page = _page
            total = json.total
        })
    }
    $: get_question_list($page)
</script>

<div class="container my-3">
    <div class="d-flex justify-content-between align-items-end mb-2">
        <h4>자유게시판</h4>
        <button class="btn btn-outline-secondary btn-sm" on:click="{() => push('/')}">🏠 홈으로</button>
    </div>

    <table class="table">
        <thead>
        <tr class="text-center table-dark">
            <th>번호</th>
            <th style="width:50%">제목</th>
            <th>글쓴이</th>
            <th>작성일시</th>
        </tr>
        </thead>
        <tbody>
        {#each question_list as question, i}
        <tr class="text-center">
            <td>{total - ($page * size)- i}</td>
            <td class="text-start">
                <a use:link href="/detail/{question.id}">{question.subject}</a>
                {#if question.answers.length > 0 }
                    <span class="text-danger small mx-2">{question.answers.length}</span>
                {/if}
            </td>
            <td>{ question.user ? question.user.username : "" }</td>
            <td>{moment(question.create_date).format("YYYY년 MM월 DD일 hh:mm a")}</td>
        </tr>
        {/each}
        </tbody>
    </table>

    <ul class="pagination justify-content-center">
        <li class="page-item {$page <= 0 && 'disabled'}">
            <button class="page-link" on:click="{() => get_question_list($page-1)}">이전</button>
        </li>
        {#each Array(total_page) as _, loop_page}
        {#if loop_page >= $page-5 && loop_page <= $page+5}
        <li class="page-item {loop_page === $page && 'active'}">
            <button on:click="{() => get_question_list(loop_page)}" class="page-link">{loop_page+1}</button>
        </li>
        {/if}
        {/each}
        <li class="page-item {$page >= total_page-1 && 'disabled'}">
            <button class="page-link" on:click="{() => get_question_list($page+1)}">다음</button>
        </li>
    </ul>
    <div class="d-flex justify-content-between mt-3">
        <button class="btn btn-secondary" on:click="{() => push('/')}">⬅️ 홈으로</button>
        <a use:link href="/question-create" class="btn btn-primary {$is_login ? '' : 'disabled'}">질문 등록하기</a>
    </div>
</div>