import cv2
import cv2.aruco as aruco
import numpy as np
import json
import time
import pyautogui
import torch
import sys
import os
from ultralytics import YOLO

# 라이브러리 경로 (DA3용)
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

from depth_anything_3.api import DepthAnything3

# ==========================================
# 0. GPU 설정
# ==========================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 실행 디바이스: {device}")

# ==========================================
# 1. 설정
# ==========================================
WARP_W = 1200
WARP_H = 620
LAYOUT_FILE = "key_layout.json"
YOLO_PATH = R'finger_project/finger_project/train_result/weights/YUN_best.pt'
DA3_MODEL_ID = "depth-anything/DA3-Small"

# [튜닝 파라미터]
STOP_SPEED_THRESHOLD = 50.0  # 이 속도보다 느리면 '멈춤' (픽셀/초)
MIN_STOP_DURATION = 0.05  # 이 시간 이상 멈춰야 함 (초)
COOLDOWN_TIME = 0.2  # 중복 입력 방지 (초)
TOUCH_DEPTH_THRESHOLD = 15  # (손깊이 - 종이깊이) 차이가 이보다 작아야 'Touch' (DA3 필터)

# 특수 키 매핑
SPECIAL_KEYS = {
    "SpaceBar": "space", "Enter": "enter", "Backspace": "backspace",
    "Tab": "tab", "CapsRock": "capslock", "Shift": "shift",
    "RShift": "shiftright", "Ctrl": "ctrl", "Win": "win",
    "Alt": "alt", "up": "up", "down": "down",
    "left": "left", "right": "right", "~": "`"
}

# ==========================================
# 2. 초기화 (모델 로드)
# ==========================================
# JSON 로드
try:
    with open(LAYOUT_FILE, "r", encoding='utf-8') as f:
        raw_layout = json.load(f)
    KEY_LAYOUT = {}
    for k, v in raw_layout.items():
        KEY_LAYOUT[k] = {'x': v[0], 'y': v[1], 'w': v[2], 'h': v[3]}
except FileNotFoundError:
    print("❌ key_layout.json 파일 없음")
    exit()

# YOLO & ArUco
model = YOLO(YOLO_PATH)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_25h9)
parameters = aruco.DetectorParameters()

# DA3 로드
try:
    da3_model = DepthAnything3.from_pretrained(DA3_MODEL_ID).to(device).eval()
except:
    print("❌ DA3 모델 로드 실패")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened(): exit()

# 상태 변수 { id : { ... } }
fingers_state = {}

print("=== 하이브리드(Velocity + Depth) 키보드 시작 (종료: q) ===")

while True:
    ret, frame = cap.read()
    if not ret: break

    # ----------------------------------------------------
    # [1] 환경 인식 (마커 & Depth)
    # ----------------------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    matrix = None
    warped_view = np.zeros((WARP_H, WARP_W, 3), dtype=np.uint8)
    depth_uint8 = None
    paper_z = 0

    if ids is not None and len(ids) >= 4:
        ids = ids.flatten()
        corners_map = {id: corner for id, corner in zip(ids, corners)}

        if all(i in corners_map for i in [0, 1, 2, 3]):
            try:
                src_pts = np.array([
                    corners_map[0][0][1], corners_map[1][0][0],
                    corners_map[3][0][3], corners_map[2][0][2]
                ], dtype=np.float32)
                dst_pts = np.array([
                    [0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]
                ], dtype=np.float32)

                matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

                # DA3 추론용 이미지 (Warped)
                warped_view_da3 = cv2.warpPerspective(frame, matrix, (WARP_W, WARP_H))

                # DA3 추론 (가상 화면 전체 깊이 계산)
                # (GPU 가속 덕분에 Small 모델은 실시간 가능)
                da3_res = da3_model.inference([warped_view_da3])
                depth_map = da3_res.depth[0]

                # 정규화 및 리사이즈 (WARP 크기에 맞춤)
                d_min, d_max = depth_map.min(), depth_map.max()
                depth_norm = (depth_map - d_min) / (d_max - d_min)
                depth_uint8 = (depth_norm * 255).astype(np.uint8)

                if depth_uint8.shape[:2] != (WARP_H, WARP_W):
                    depth_uint8 = cv2.resize(depth_uint8, (WARP_W, WARP_H))

                # 종이 깊이(paper_z) 계산: 네 모서리 평균
                margin = 20
                h, w = depth_uint8.shape
                corners_roi = np.concatenate([
                    depth_uint8[0:margin, 0:margin].flatten(),
                    depth_uint8[0:margin, w - margin:w].flatten(),
                    depth_uint8[h - margin:h, w - margin:w].flatten(),
                    depth_uint8[h - margin:h, 0:margin].flatten()
                ])
                paper_z = int(np.mean(corners_roi))

                aruco.drawDetectedMarkers(frame, corners, ids)
            except:
                pass

    # 키보드 그리기 (가상 화면)
    for key_name, rect in KEY_LAYOUT.items():
        rx, ry, rw, rh = rect['x'], rect['y'], rect['w'], rect['h']
        cv2.rectangle(warped_view, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 1)
        cv2.putText(warped_view, key_name, (rx + 5, ry + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    # ----------------------------------------------------
    # [2] 손가락 추적 및 하이브리드 판정
    # ----------------------------------------------------
    results = model.track(frame, persist=True, verbose=False, device=device)
    curr_time = time.time()
    current_ids = set()

    for r in results:
        if r.boxes.id is None: continue
        boxes = r.boxes.xyxy.cpu().numpy()
        track_ids = r.boxes.id.int().cpu().numpy()

        for box, track_id in zip(boxes, track_ids):
            current_ids.add(track_id)

            if track_id not in fingers_state:
                fingers_state[track_id] = {
                    'prev_pos': None, 'prev_time': 0, 'hover_key': None,
                    'stop_time': 0, 'last_input': 0
                }
            st = fingers_state[track_id]

            # 좌표 추출
            x1, y1, x2, y2 = map(int, box)
            fx = (x1 + x2) / 2
            fy = y2 * 0.9 + y1 * 0.1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if matrix is not None and depth_uint8 is not None:
                pts = np.array([[[fx, fy]]], dtype=np.float32)
                trans = cv2.perspectiveTransform(pts, matrix)
                tx, ty = trans[0][0]

                # 범위 체크
                if 0 <= tx < WARP_W and 0 <= ty < WARP_H:
                    # 1. 속도 계산 (Velocity)
                    speed = 9999
                    if st['prev_pos'] is not None and st['prev_time'] > 0:
                        dt = curr_time - st['prev_time']
                        if dt > 0:
                            dist = np.linalg.norm(np.array([tx, ty]) - np.array(st['prev_pos']))
                            speed = dist / dt  # px/sec

                    # 2. 깊이 계산 (Depth)
                    finger_z = depth_uint8[int(ty), int(tx)]
                    depth_diff = int(finger_z) - int(paper_z)

                    # 시각화 (손 위치 및 깊이 정보)
                    cv2.circle(warped_view, (int(tx), int(ty)), 8, (0, 0, 255), -1)
                    cv2.putText(warped_view, f"S:{speed:.0f} D:{depth_diff}", (int(tx), int(ty) - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # 키 히트 테스트
                    detected_key = None
                    for key_name, rect in KEY_LAYOUT.items():
                        rx, ry, rw, rh = rect['x'], rect['y'], rect['w'], rect['h']
                        if rx < tx < rx + rw and ry < ty < ry + rh:
                            detected_key = key_name
                            break

                    # ==================================================
                    # ★ [핵심] 하이브리드 입력 판정 로직 ★
                    # 조건: (속도가 멈춤) AND (깊이가 낮음)
                    # ==================================================
                    is_stopped = speed < STOP_SPEED_THRESHOLD
                    is_low = depth_diff < TOUCH_DEPTH_THRESHOLD  # 차이가 작을수록(음수포함) 바닥에 가까움

                    if detected_key:
                        # 키 위에 있고 멈췄는가?
                        if is_stopped:
                            if st['hover_key'] != detected_key:
                                st['hover_key'] = detected_key
                                st['stop_time'] = curr_time

                            elif (curr_time - st['stop_time']) > MIN_STOP_DURATION:
                                if (curr_time - st['last_input']) > COOLDOWN_TIME:

                                    # ★ 마지막 관문: 깊이 체크 (DA3 필터) ★
                                    if is_low:
                                        print(f"👉 Touch(ID:{track_id}): {detected_key} (Diff:{depth_diff})")

                                        py_key = SPECIAL_KEYS.get(detected_key, detected_key.lower())
                                        if py_key: pyautogui.press(py_key)

                                        st['last_input'] = curr_time
                                        # 입력 성공 시각화
                                        rx, ry, rw, rh = KEY_LAYOUT[detected_key]['x'], KEY_LAYOUT[detected_key]['y'], \
                                        KEY_LAYOUT[detected_key]['w'], KEY_LAYOUT[detected_key]['h']
                                        cv2.rectangle(warped_view, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), -1)
                                    else:
                                        # 멈췄지만 높이가 높음 -> Hovering 상태
                                        rx, ry, rw, rh = KEY_LAYOUT[detected_key]['x'], KEY_LAYOUT[detected_key]['y'], \
                                        KEY_LAYOUT[detected_key]['w'], KEY_LAYOUT[detected_key]['h']
                                        cv2.rectangle(warped_view, (rx, ry), (rx + rw, ry + rh), (0, 255, 255),
                                                      2)  # 노란 테두리
                        else:
                            st['stop_time'] = curr_time  # 움직이면 리셋
                    else:
                        st['hover_key'] = None
                        st['stop_time'] = 0

                    # 상태 업데이트
                    st['prev_pos'] = (tx, ty)
                    st['prev_time'] = curr_time

    # ID 정리
    expired_ids = [k for k in fingers_state.keys() if k not in current_ids]
    for k in expired_ids: del fingers_state[k]

    cv2.imshow("Tracking Cam", frame)
    cv2.imshow("Hybrid Keyboard", warped_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()