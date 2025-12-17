<script>
    import fastapi from "../lib/api"
    import { link, push } from 'svelte-spa-router'
    import { page, is_login } from "../lib/store"
    import moment from "moment/min/moment-with-locales"
    moment.locale('ko')

    let question_list = []
    let size = 9 // [변경] 카드형은 3개씩 보기 좋게 9개로 변경
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

<div class="container my-4">
    <div class="d-flex justify-content-between align-items-center mb-4">
        <div>
            <h2 class="fw-bold m-0">🗣️ 자유게시판</h2>
            <small class="text-muted">편하게 이야기를 나누는 공간입니다.</small>
        </div>
        <div class="d-flex gap-2">
            <button class="btn btn-outline-secondary" on:click="{() => push('/')}">🏠 홈</button>
            <a use:link href="/question-create" class="btn btn-primary {$is_login ? '' : 'disabled'}">
                ✏️ 글쓰기
            </a>
        </div>
    </div>

    <div class="row row-cols-1 row-cols-md-2 row-cols-lg-3 g-4">
        {#each question_list as question}
        <div class="col">
            <div class="card h-100 shadow-sm hover-card" on:click="{() => push('/detail/' + question.id)}">
                <div class="card-body d-flex flex-column">
                    <h5 class="card-title fw-bold text-truncate mb-3">
                        {question.subject}
                    </h5>

                    <div class="mt-auto">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <small class="text-primary fw-bold">
                                {question.user ? question.user.username : "익명"}
                            </small>
                            <small class="text-muted" style="font-size: 0.8rem;">
                                {moment(question.create_date).fromNow()}
                            </small>
                        </div>

                        <div class="d-flex gap-3 text-secondary small border-top pt-2">
                            <span>
                                💬 댓글 {question.answers.length}
                            </span>
                            </div>
                    </div>
                </div>
            </div>
        </div>
        {/each}
    </div>

    <ul class="pagination justify-content-center mt-5">
        <li class="page-item {$page <= 0 && 'disabled'}">
            <button class="page-link rounded-pill px-3 mx-1" on:click="{() => get_question_list($page-1)}">이전</button>
        </li>
        {#each Array(total_page) as _, loop_page}
        {#if loop_page >= $page-5 && loop_page <= $page+5}
        <li class="page-item {loop_page === $page && 'active'}">
            <button on:click="{() => get_question_list(loop_page)}" class="page-link rounded-circle mx-1" style="width: 40px; height: 40px;">{loop_page+1}</button>
        </li>
        {/if}
        {/each}
        <li class="page-item {$page >= total_page-1 && 'disabled'}">
            <button class="page-link rounded-pill px-3 mx-1" on:click="{() => get_question_list($page+1)}">다음</button>
        </li>
    </ul>
</div>

<style>
    /* 마우스 올렸을 때 살짝 떠오르는 효과 */
    .hover-card {
        transition: transform 0.2s, box-shadow 0.2s;
        cursor: pointer;
        border: none;
        background-color: #ffffff;
    }
    .hover-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1) !important;
    }
    .card-title {
        color: #333;
    }

    /* 페이징 버튼 예쁘게 */
    .page-link {
        color: #333;
        border: none;
        background-color: #f8f9fa;
    }
    .page-item.active .page-link {
        background-color: #0d6efd;
        color: white;
        font-weight: bold;
    }
</style>