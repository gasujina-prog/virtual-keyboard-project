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
    print("🚀 질문 등록 요청 들어옴!")  # 디버깅용 로그

    data = request.get_json()
    subject = data.get('subject')
    content = data.get('content')
    username = data.get('username')

    if not subject or not content:
        return jsonify({'detail': '제목과 내용을 입력해주세요.'}), 400

    # 사용자 조회
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({'detail': '존재하지 않는 사용자입니다.'}), 404

    # DB 저장
    q = Question(subject=subject, content=content, create_date=datetime.now(), user=user)
    db.session.add(q)
    db.session.commit()

    return jsonify({'message': '게시글이 등록되었습니다.'}), 201

# ------------------------------------------------------
# 4. 질문 수정 (Update)
# ------------------------------------------------------
@bp.route('/question/modify/<int:question_id>', methods=['PUT'])  # 혹은 POST
@bp.route('/v1/question/modify/<int:question_id>', methods=['PUT'])
def question_modify(question_id):
    print(f"🛠️ 게시글 수정 요청: ID {question_id}")

    data = request.get_json()
    username = data.get('username')
    subject = data.get('subject')
    content = data.get('content')

    # 1. 게시글 찾기
    question = Question.query.get_or_404(question_id)

    # 2. 권한 확인 (본인 확인)
    # (실제 서비스에선 토큰으로 하지만, 지금은 username으로 약식 체크)
    if question.user.username != username:
        print(f"❌ 수정 권한 없음: 작성자({question.user.username}) != 요청자({username})")
        return jsonify({'detail': '수정 권한이 없습니다.'}), 403

    # 3. 데이터 수정
    question.subject = subject
    question.content = content
    # question.modify_date = datetime.now() # 모델에 컬럼이 있다면 추가

    db.session.commit()
    print(f"✅ 게시글 수정 완료: {subject}")
    return jsonify({'message': '게시글이 수정되었습니다.'})

# ------------------------------------------------------
# 5. 질문 삭제 (Delete)
# ------------------------------------------------------
@bp.route('/question/delete/<int:question_id>', methods=['DELETE'])  # 혹은 POST
@bp.route('/v1/question/delete/<int:question_id>', methods=['DELETE'])
def question_delete(question_id):
    print(f"🗑️ 게시글 삭제 요청: ID {question_id}")

    # (삭제 요청 시에는 body에 username을 담아 보내거나, 쿼리 파라미터로 받아야 함)
    # 여기서는 간단히 JSON으로 받는다고 가정
    data = request.get_json() or {}
    username = data.get('username')

    question = Question.query.get_or_404(question_id)

    # 권한 확인
    if question.user.username != username:
        print(f"❌ 삭제 권한 없음: 작성자({question.user.username}) != 요청자({username})")
        return jsonify({'detail': '삭제 권한이 없습니다.'}), 403

    db.session.delete(question)
    db.session.commit()
    print(f"✅ 게시글 삭제 완료: ID {question_id}")
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
# 8. 답변 삭제 (Delete Answer)
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
# 9. 답변 추천 (Vote Answer)
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