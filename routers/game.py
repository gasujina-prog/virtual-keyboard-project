from flask import Blueprint, request, jsonify, session
from core.database import db
from models.game_result import GameResult
from models.user import User
from datetime import datetime

# Blueprint 설정 (URL prefix: /game)
game_bp = Blueprint('game', __name__, url_prefix='/game')


# ---------------------------------------------------------
# API: 게임 결과 저장
# 요청 URL: POST /game/save
# 요청 데이터: { "score": 1500, "accuracy": 95.5, "mode": "Normal" }
# ---------------------------------------------------------
@game_bp.route('/save', methods=['POST'])
def save_game_result():
    # 1. 로그인 여부 확인 (세션에 user_id가 있는지 체크)
    user_id = session.get('user_id')

    # 디버깅용: 현재 세션 상태 출력 (나중에 주석 처리 가능)
    print(f"DEBUG: Session User ID: {user_id}")

    if not user_id:
        return jsonify({'result': 'fail', 'message': '로그인이 필요합니다.'}), 401

    # 2. 프론트엔드에서 보낸 데이터 받기
    data = request.get_json()
    score = data.get('score')
    accuracy = data.get('accuracy')
    mode = data.get('mode')

    # 데이터 유효성 검사
    if score is None or accuracy is None or mode is None:
        return jsonify({'result': 'fail', 'message': '데이터가 부족합니다.'}), 400

    try:
        # 3. DB 모델 객체 생성
        new_result = GameResult(
            score=score,
            accuracy=accuracy,
            mode=mode,
            create_date=datetime.now(),
            user_id=user_id
        )

        # 4. 저장 (Commit)
        db.session.add(new_result)
        db.session.commit()

        print(f"✅ 게임 결과 저장 완료: {score}점 ({mode})")
        return jsonify({'result': 'success', 'message': '저장되었습니다!', 'id': new_result.id}), 201

    except Exception as e:
        db.session.rollback()
        print(f"❌ 게임 저장 에러: {e}")
        return jsonify({'result': 'fail', 'message': '서버 오류 발생'}), 500


# [추가] API: 내 게임 통계 조회 (마이페이지용)
# 요청 URL: GET /game/stats
# ---------------------------------------------------------
@game_bp.route('/stats', methods=['GET'])
def get_stats():
    user_id = session.get('user_id')

    if not user_id:
        return jsonify({'result': 'fail', 'message': '로그인 필요'}), 401

    try:
        # 1. 유저 정보 조회
        user = User.query.get(user_id)
        username = user.username if user else "Unknown"

        # 2. 해당 유저의 모든 게임 기록 조회 (최신순)
        results = GameResult.query.filter_by(user_id=user_id).order_by(GameResult.create_date.desc()).all()

        if not results:
            return jsonify({
                'result': 'success',
                'username': username,
                'total_games': 0,
                'high_score': 0,
                'avg_accuracy': 0,
                'recent_history': []
            })

        # 3. 통계 계산
        total_games = len(results)
        high_score = max([r.score for r in results])
        # 정확도 평균 계산 (소수점 반올림)
        avg_accuracy = round(sum([r.accuracy for r in results]) / total_games)

        # 4. 최근 기록 10개 데이터 가공
        history = []
        for r in results[:10]:
            history.append({
                'date': r.create_date.strftime('%Y-%m-%d %H:%M'),
                'mode': r.mode,
                'score': r.score,
                'accuracy': int(r.accuracy)
            })

        return jsonify({
            'result': 'success',
            'username': username,
            'total_games': total_games,
            'high_score': high_score,
            'avg_accuracy': avg_accuracy,
            'recent_history': history
        })

    except Exception as e:
        print(f"❌ 통계 조회 에러: {e}")
        return jsonify({'result': 'error'}), 500