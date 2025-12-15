from flask import Blueprint, request, jsonify
from models.board_model import Question

bp = Blueprint('board', __name__, url_prefix='/api')


# ------------------------------------------------------
# 1. 질문 목록 조회 (List)
# ------------------------------------------------------
@bp.route('/question/list', methods=['GET'])
@bp.route('/v1/question/list', methods=['GET'])
def question_list():
    page = request.args.get('page', type=int, default=0)
    size = request.args.get('size', type=int, default=10)

    # 작성일 기준 내림차순(최신순) 정렬
    question_list = Question.query.order_by(Question.create_date.desc())
    total = question_list.count()
    question_list = question_list.offset(page * size).limit(size).all()

    data = []
    for q in question_list:
        data.append({
            'id': q.id,
            'subject': q.subject,
            'create_date': q.create_date.isoformat(),
            'user': {'username': q.user.username} if q.user else None,
            # 목록에서는 답변 개수 정도만 보여주거나, ID만 간단히 넘김
            'answers': [{'id': a.id} for a in q.answers]
        })
    return jsonify({'total': total, 'question_list': data})


# ------------------------------------------------------
# 2. 질문 상세 조회 (Detail) - [추가된 부분]
# ------------------------------------------------------
@bp.route('/question/detail/<int:question_id>', methods=['GET'])
@bp.route('/v1/question/detail/<int:question_id>', methods=['GET'])
def question_detail(question_id):
    # 해당 ID의 질문을 찾고, 없으면 404 에러를 자동 반환
    question = Question.query.get_or_404(question_id)

    # 질문에 달린 답변들도 함께 정리해서 보냄
    answers_data = []
    for answer in question.answers:
        answers_data.append({
            'id': answer.id,
            'content': answer.content,
            'create_date': answer.create_date.isoformat(),
            'user': {'username': answer.user.username} if answer.user else None
        })

    return jsonify({
        'id': question.id,
        'subject': question.subject,
        'content': question.content,
        'create_date': question.create_date.isoformat(),
        'user': {'username': question.user.username} if question.user else None,
        'answers': answers_data
    })