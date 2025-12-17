import time
from flask import Blueprint, Response, jsonify, request
from core import state

bp = Blueprint('stream', __name__)


@bp.route('/api/keyboard/toggle', methods=['POST'])
def toggle_virtual_keyboard():
    data = request.get_json()
    active_status = data.get('active', True)

    # 전역 상태 업데이트
    state.is_virtual_input_active = active_status

    # Detector(AI)에도 알림
    if state.detector:
        state.detector.set_active(active_status)

    status_text = "ON" if active_status else "OFF"
    print(f"🎛️ Virtual Keyboard Input is now: {status_text}")
    return jsonify({"message": f"Input {status_text}"})


@bp.route('/video_feed_cam')
def video_feed_cam():
    def generate():
        while True:
            # [수정 1] 꺼져 있으면 데이터 전송 중단 (1초에 한 번만 체크하며 대기)
            if state.detector is None or not state.is_virtual_input_active:
                time.sleep(1.0)
                continue

            cam, _ = state.detector.get_frames()

            # 프레임이 아직 준비 안 됐으면 대기
            if cam is None:
                time.sleep(0.05)
                continue

            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + cam + b'\r\n')

            # [수정 2] 전송 속도 조절 (약 30FPS) -> 버벅임 제거
            time.sleep(0.03)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@bp.route('/video_feed_warp')
def video_feed_warp():
    def generate():
        while True:
            # [수정 1] 꺼져 있으면 데이터 전송 중단
            if state.detector is None or not state.is_virtual_input_active:
                time.sleep(1.0)
                continue

            _, warp = state.detector.get_frames()

            if warp is None:
                time.sleep(0.05)
                continue

            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + warp + b'\r\n')

            # [수정 2] 전송 속도 조절
            time.sleep(0.03)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@bp.route('/stream')
def stream():
    def event_stream():
        while True:
            # SSE 큐에서 키 입력 가져오기
            try:
                # timeout을 줘서 무한 대기 방지 (서버 종료 시 빠져나오기 위함)
                key = state.sse_queue.get(timeout=1.0)
                yield f"data: {key}\n\n"
            except:
                # 큐가 비어있으면(timeout) 그냥 루프 다시 돌기
                continue

    return Response(event_stream(), mimetype="text/event-stream")