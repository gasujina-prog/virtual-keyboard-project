from flask import Flask, render_template, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import time
import threading

# 파일명: web_converged.py 로 변경된 것 반영
from web_converged import KeyboardDetector

app = Flask(__name__)

# ★ DB 이름 변경 (확장성을 위해) ★
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///web_project.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# DB 모델 (키보드 로그)
class KeyLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key_name = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        return {
            "id": self.id,
            "key": self.key_name,
            "time": self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }


# 비전 객체
detector = KeyboardDetector()


# DB 저장 스레드
def save_keys_worker():
    print("[INFO] DB 저장 워커 가동")
    while True:
        time.sleep(1)  # 1초 단위로 저장
        inputs = detector.pop_inputs()
        if inputs:
            with app.app_context():
                for item in inputs:
                    new_log = KeyLog(key_name=item['key'])
                    db.session.add(new_log)
                db.session.commit()
                # print(f"💾 Saved {len(inputs)} keys") # 로그 너무 많으면 주석 처리


@app.route('/')
def index():
    return render_template('index.html')


# ★ 프론트엔드가 데이터를 요청할 주소 (API) ★
@app.route('/api/logs')
def get_logs():
    # 최신순으로 10개만 가져오기
    logs = KeyLog.query.order_by(KeyLog.id.desc()).limit(10).all()
    return jsonify([log.to_dict() for log in logs])


@app.route('/video_feed_cam')
def video_feed_cam():
    def generate():
        while True:
            cam, _ = detector.get_frames()
            if cam: yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + cam + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_warp')
def video_feed_warp():
    def generate():
        while True:
            _, warp = detector.get_frames()
            if warp: yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + warp + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    try:
        with app.app_context():
            db.create_all()

        detector.start()

        t = threading.Thread(target=save_keys_worker, daemon=True)
        t.start()

        print("[INFO] 서버 시작: http://127.0.0.1:5000")
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        pass
    finally:
        detector.stop()