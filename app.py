from flask import Flask, render_template, Response
import time
# web_converged.py 파일에서 KeyboardDetector 클래스를 가져옵니다.
from web_converged import KeyboardDetector

app = Flask(__name__,
            static_folder='static',
            template_folder='templates',
            static_url_path='/')

# 1. 감지기 객체 생성 및 시작
# 서버가 켜질 때 딱 한 번만 실행됩니다.
detector = KeyboardDetector()
detector.start()

# 2. 영상 스트리밍을 위한 제너레이터 함수
def generate_frames(frame_type):
    while True:
        # web_converged에서 최신 프레임 두 개를 가져옵니다.
        frame_cam, frame_warp = detector.get_frames()

        target_frame = None
        if frame_type == 'cam':
            target_frame = frame_cam
        elif frame_type == 'warp':
            target_frame = frame_warp

        # 아직 프레임이 준비 안 됐으면 잠시 대기
        if target_frame is None:
            time.sleep(0.01)
            continue

        # 3. 웹 브라우저가 이해하는 MJPEG 포맷으로 전송
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + target_frame + b'\r\n')

# --- 라우트 설정 ---

@app.route('/')
def home():
    return render_template('index.html')

# 카메라 원본 영상 주소
@app.route('/video_feed_cam')
def video_feed_cam():
    return Response(generate_frames('cam'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 가상 키보드 영상 주소
@app.route('/video_feed_warp')
def video_feed_warp():
    return Response(generate_frames('warp'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Svelte 라우팅 처리
@app.route('/<path:path>')
def catch_all(path):
    return render_template('index.html')

if __name__ == '__main__':
    try:
        print("⚡ 서버 시작: http://127.0.0.1:5000")
        app.run(debug=True, port=5000, use_reloader=False)
        # use_reloader=False: 카메라가 두 번 켜지는 것을 방지
    finally:
        detector.stop() # 서버 꺼질 때 카메라 정리