import os
import secrets
import threading
from flask import Flask
from flask_cors import CORS

# 모듈 가져오기
from core.database import db
from core import state
from services.web_converged import KeyboardDetector
from services.worker import save_keys_worker

# [수정] 라우터 가져오기 (board_router만 이름 변경됨)
from routers import auth, board_router, stream, views

# ==========================================
# 1. Flask 앱 설정
# ==========================================
app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

CORS(app)

# DB 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FOLDER = "database"
DB_DIR = os.path.join(BASE_DIR, DB_FOLDER)
if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)

db_path = os.path.join(DB_DIR, "web_project.db")
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = secrets.token_hex(16)

# DB 초기화
db.init_app(app)

# ==========================================
# 2. 블루프린트 등록 (라우터 연결)
# ==========================================
app.register_blueprint(auth.bp)
app.register_blueprint(board_router.bp)   # [중요] board -> board_router
app.register_blueprint(stream.bp)
app.register_blueprint(views.bp)

# ==========================================
# 3. 서버 실행 및 백그라운드 작업 시작
# ==========================================
if __name__ == "__main__":
    # DB 테이블 생성
    with app.app_context():
        # [중요] models를 import해야 테이블이 생성됨 (이름 변경 반영)
        from models import user, board_model, key_log
        db.create_all()
        print(f"✅ Database ready at: {db_path}")

    # Vision Engine 시작
    state.detector = KeyboardDetector()
    state.detector.start()

    # 백그라운드 Worker 시작
    t = threading.Thread(target=save_keys_worker, args=(app,), daemon=True)
    t.start()

    print("⚡ Server started at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)