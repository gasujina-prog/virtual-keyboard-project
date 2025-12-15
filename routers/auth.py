import secrets
from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from core.database import db
from core import state
from models.user import User

bp = Blueprint('auth', __name__, url_prefix='/api')

@bp.route('/user/create', methods=['POST'])
@bp.route('/v1/user/create', methods=['POST'])
def user_create():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password1') or data.get('password')

    if not username or not password or not email:
        return jsonify({"detail": "모든 항목을 입력해주세요."}), 400

    user = User.query.filter((User.username == username) | (User.email == email)).first()
    if user:
        return jsonify({"detail": "이미 등록된 사용자 또는 이메일입니다."}), 409

    hashed_pw = generate_password_hash(password)
    new_user = User(username=username, password=hashed_pw, email=email)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "회원가입 성공"}), 201

@bp.route('/user/login', methods=['POST'])
@bp.route('/v1/user/login', methods=['POST'])
def user_login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()

    if not user or not check_password_hash(user.password, password):
        return jsonify({"detail": "아이디 또는 비밀번호가 올바르지 않습니다."}), 401

    state.current_user_id = user.id
    print(f"🔑 User Login: {user.username} (ID: {user.id})")

    return jsonify({
        "access_token": secrets.token_hex(16),
        "username": user.username,
        "token_type": "bearer"
    })

@bp.route('/user/logout', methods=['POST'])
def user_logout():
    state.current_user_id = None
    print("🔒 User Logout")
    return jsonify({"message": "로그아웃 성공"})