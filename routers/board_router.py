from datetime import datetime
from flask import Blueprint, request, jsonify
from core.database import db
from core import state
from models.board_model import Question, Answer
from models.user import User

# URL 접두사 설정 (/api)
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
            'answers': [{'id': a.id} for a in q.answers]
        })
    return jsonify({'total': total, 'question_list': data})


# ------------------------------------------------------
# 2. 질문 상세 조회 (Detail)
# ------------------------------------------------------
@bp.route('/question/detail/<int:question_id>', methods=['GET'])
@bp.route('/v1/question/detail/<int:question_id>', methods=['GET'])
def question_detail(question_id):
    question = Question.query.get_or_404(question_id)

    answers_data = []
    for answer in question.answers:
        answers_data.append({
            'id': answer.id,
            'content': answer.content,
            'create_date': answer.create_date.isoformat(),
            'user': {'username': answer.user.username} if answer.user else None,
            'voter': [{'id': v.id, 'username': v.username} for v in answer.voter]
        })

    return jsonify({
        'id': question.id,
        'subject': question.subject,
        'content': question.content,
        'create_date': question.create_date.isoformat(),
        'user': {'username': question.user.username} if question.user else None,
        'answers': answers_data,
        'voter': [{'id': v.id, 'username': v.username} for v in question.voter]
    })

# ------------------------------------------------------
# 3. 질문 등록 (Create)
# ------------------------------------------------------
@bp.route('/question/create', methods=['POST'])
@bp.route('/v1/question/create', methods=['POST'])
def question_create():
    # 1. 로그인 확인
    if state.current_user_id is None:
        return jsonify({'detail': '로그인이 필요합니다.'}), 401

    data = request.get_json()
    subject = data.get('subject')
    content = data.get('content')

    if not subject or not content:
        return jsonify({'detail': '제목과 내용을 입력해주세요.'}), 400

    # 2. 작성자 할당 (프론트에서 보낸 username 무시하고, 실제 로그인한 유저 찾기)
    user = User.query.get(state.current_user_id)

    # DB저장
    q = Question(subject=subject, content=content, create_date=datetime.now(), user=user)
    db.session.add(q)
    db.session.commit()

    return jsonify({'message': '게시글이 등록되었습니다.'}), 201

# ------------------------------------------------------
# 4. 질문 수정 (Update)
# ------------------------------------------------------
@bp.route('/question/modify/<int:question_id>', methods=['PUT'])
@bp.route('/v1/question/modify/<int:question_id>', methods=['PUT'])
def question_modify(question_id):
    # 1. 로그인 확인
    if state.current_user_id is None:
        return jsonify({'detail': '로그인이 필요합니다.'}), 401

    data = request.get_json()
    subject = data.get('subject')
    content = data.get('content')

    question = Question.query.get_or_404(question_id)

    # 2. 권한 확인 (작성자 ID와 현재 로그인 ID 비교)
    if question.user.id != state.current_user_id:
        return jsonify({'detail': '수정 권한이 없습니다.'}), 403

    question.subject = subject
    question.content = content
    db.session.commit()

    return jsonify({'message': '게시글이 수정되었습니다.'})

# ------------------------------------------------------
# 5. 질문 삭제 (Delete)
# ------------------------------------------------------
@bp.route('/question/delete/<int:question_id>', methods=['DELETE'])
@bp.route('/v1/question/delete/<int:question_id>', methods=['DELETE'])
def question_delete(question_id):
    # 1. 로그인 확인
    if state.current_user_id is None:
        return jsonify({'detail': '로그인이 필요합니다.'}), 401

    question = Question.query.get_or_404(question_id)

    # 2. 권한 확인
    if question.user.id != state.current_user_id:
        return jsonify({'detail': '삭제 권한이 없습니다.'}), 403

    db.session.delete(question)
    db.session.commit()

    return jsonify({'message': '게시글이 삭제되었습니다.'})

# ------------------------------------------------------
# 6. 질문 추천 (Vote) - [새로 추가]
# ------------------------------------------------------
@bp.route('/question/vote', methods=['POST'])
@bp.route('/v1/question/vote', methods=['POST'])
def question_vote():
    data = request.get_json()
    question_id = data.get('question_id')

    # 1. 질문 찾기
    question = Question.query.get_or_404(question_id)

    # 2. 로그인 사용자 확인 (state.current_user_id 사용)
    if state.current_user_id is None:
        return jsonify({'detail': '로그인이 필요합니다.'}), 401

    user = User.query.get(state.current_user_id)

    # 3. 본인 글 추천 방지 (선택 사항)
    if question.user_id == user.id:
        return jsonify({'detail': '본인이 작성한 글은 추천할 수 없습니다.'}), 400

    # 4. 이미 추천했는지 확인 후 토글(추천/취소) 또는 추가
    if user in question.voter:
        # 이미 추천했다면 추천 취소할지, 아니면 중복 금지라고 할지 결정
        # 여기서는 "이미 추천했습니다" 에러를 띄웁니다.
        return jsonify({'detail': '이미 추천한 게시글입니다.'}), 409
    else:
        question.voter.append(user)
        db.session.commit()

    return jsonify({'message': '추천 완료'})

# ------------------------------------------------------
# 7. 답변 등록 (Create Answer)
# ------------------------------------------------------
@bp.route('/answer/create/<int:question_id>', methods=['POST'])
@bp.route('/v1/answer/create/<int:question_id>', methods=['POST'])
def answer_create(question_id):
    question = Question.query.get_or_404(question_id)

    if state.current_user_id is None:
        return jsonify({'detail': '로그인이 필요합니다.'}), 401

    user = User.query.get(state.current_user_id)
    data = request.get_json()
    content = data.get('content')

    if not content:
        return jsonify({'detail': '내용을 입력해주세요.'}), 400

    new_answer = Answer(question=question, content=content, user=user, create_date=datetime.now())

    db.session.add(new_answer)
    db.session.commit()

    return jsonify({'message': '답변 등록 성공'})

# ------------------------------------------------------
# 8. 답변 수정 (Update Answer)
# ------------------------------------------------------
@bp.route('/answer/update', methods=['PUT'])
@bp.route('/v1/answer/update', methods=['PUT'])
def answer_modify():
    if state.current_user_id is None:
        return jsonify({'detail': '로그인이 필요합니다.'}), 401

    # 프론트엔드가 JSON body에 'answer_id'를 담아서 보냅니다.
    data = request.get_json()
    answer_id = data.get('answer_id')
    content = data.get('content')

    answer = Answer.query.get_or_404(answer_id)

    # 권한 확인 (본인만 수정 가능)
    if state.current_user_id != answer.user.id:
        return jsonify({'detail': '수정 권한이 없습니다.'}), 403

    answer.content = content
    db.session.commit()

    return jsonify({'message': '답변이 수정되었습니다.'})
# ------------------------------------------------------
# 9. 답변 삭제 (Delete Answer)
# ------------------------------------------------------
@bp.route('/answer/delete', methods=['DELETE'])
@bp.route('/v1/answer/delete', methods=['DELETE'])
def answer_delete():
    if state.current_user_id is None:
        return jsonify({'detail': '로그인이 필요합니다.'}), 401

    data = request.get_json()
    answer_id = data.get('answer_id')
    answer = Answer.query.get_or_404(answer_id)
    user = User.query.get(state.current_user_id)

    if state.current_user_id != answer.user.id:
        return jsonify({'detail': '삭제 권한이 없습니다.'}), 403

    db.session.delete(answer)
    db.session.commit()

    return jsonify({'message': '삭제되었습니다.'})


# ------------------------------------------------------
# 10. 답변 추천 (Vote Answer)
# ------------------------------------------------------
@bp.route('/answer/vote', methods=['POST'])
@bp.route('/v1/answer/vote', methods=['POST'])
def answer_vote():
    if state.current_user_id is None:
        return jsonify({'detail': '로그인이 필요합니다.'}), 401

    data = request.get_json()
    answer_id = data.get('answer_id')
    answer = Answer.query.get_or_404(answer_id)
    user = User.query.get(state.current_user_id)

    # 1. 본인 추천 방지
    if answer.user_id == user.id:
        return jsonify({'detail': '본인이 작성한 글은 추천할 수 없습니다.'}), 400

    # 2. 중복 추천 방지
    if user in answer.voter:
        return jsonify({'detail': '이미 추천한 댓글입니다.'}), 409

    # 3. 추천 저장
    answer.voter.append(user)
    db.session.commit()

    return jsonify({'message': '추천 완료'})

# ------------------------------------------------------
# [추가됨] 11. 답변 상세 조회 (Detail Answer)
# 프론트엔드 AnswerModify.svelte가 호출하는 API
# ------------------------------------------------------
@bp.route('/answer/detail/<int:answer_id>', methods=['GET'])
@bp.route('/v1/answer/detail/<int:answer_id>', methods=['GET'])
def answer_detail(answer_id):
    answer = Answer.query.get_or_404(answer_id)
    return jsonify({
        'id': answer.id,
        'question_id': answer.question_id,
        'content': answer.content,
        'create_date': answer.create_date.isoformat(),
        'user': {'username': answer.user.username} if answer.user else None
    })