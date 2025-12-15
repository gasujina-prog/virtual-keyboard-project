import time
from flask import Blueprint, Response, jsonify, request
from core import state

bp = Blueprint('stream', __name__)


@bp.route('/api/keyboard/toggle', methods=['POST'])
def toggle_virtual_keyboard():
    data = request.get_json()
    active_status = data.get('active', True)

    state.is_virtual_input_active = active_status
    if state.detector:
        state.detector.set_active(active_status)

    status_text = "ON" if active_status else "OFF"
    print(f"🎛️ Virtual Keyboard Input is now: {status_text}")
    return jsonify({"message": f"Input {status_text}"})


@bp.route('/video_feed_cam')
def video_feed_cam():
    def generate():
        while True:
            if state.detector is None: time.sleep(0.1); continue
            cam, _ = state.detector.get_frames()
            if cam is None: time.sleep(0.01); continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + cam + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@bp.route('/video_feed_warp')
def video_feed_warp():
    def generate():
        while True:
            if state.detector is None: time.sleep(0.1); continue
            _, warp = state.detector.get_frames()
            if warp is None: time.sleep(0.01); continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + warp + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@bp.route('/stream')
def stream():
    def event_stream():
        while True:
            key = state.sse_queue.get()
            yield f"data: {key}\n\n"

    return Response(event_stream(), mimetype="text/event-stream")