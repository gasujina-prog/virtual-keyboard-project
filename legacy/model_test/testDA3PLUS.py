import cv2
import cv2.aruco as aruco
import numpy as np
import time
import torch
import sys
import os
import collections
import pyautogui
from ultralytics import YOLO
import json


# 라이브러리 경로
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

from depth_anything_3.api import DepthAnything3

# ==========================================
# 1. 설정
# ==========================================
WARP_W = 600  # 해상도 줄임 (속도 향상)
WARP_H = 310
LAYOUT_FILE = "key_layout.json"
YOLO_PATH = r'finger_project/finger_project/train_result/weights/YUN_best.pt'
DA3_MODEL_ID = "depth-anything/DA3-Small"

# [최적화 파라미터]
DEPTH_SKIP_FRAMES = 3  # DA3 추론을 3프레임에 1번만 수행 (속도 3배↑)
HISTORY_LEN = 5  # 노이즈 필터링용 버퍼 크기
TOUCH_DEPTH_DIFF = 10  # 터치 민감도
STOP_SPEED_THRESHOLD = 80.0

# 특수 키 매핑
SPECIAL_KEYS = {
    "SpaceBar": "space", "Enter": "enter", "Backspace": "backspace",
    "Tab": "tab", "CapsRock": "capslock", "Shift": "shift",
    "RShift": "shiftright", "Ctrl": "ctrl", "Win": "win",
    "Alt": "alt", "up": "up", "down": "down",
    "left": "left", "right": "right", "~": "`"
}

# ==========================================
# 2. 초기화
# ==========================================
try:
    with open(LAYOUT_FILE, "r", encoding='utf-8') as f:
        raw_layout = json.load(f)
    KEY_LAYOUT = {}
    # 해상도가 바뀌었으므로(600x310) 좌표도 비율에 맞게 줄여야 함
    # (기존 json이 1200x620 기준이라면 0.5배 해야 함)
    SCALE_FACTOR = 0.5
    for k, v in raw_layout.items():
        KEY_LAYOUT[k] = {
            'x': int(v[0] * SCALE_FACTOR), 'y': int(v[1] * SCALE_FACTOR),
            'w': int(v[2] * SCALE_FACTOR), 'h': int(v[3] * SCALE_FACTOR)
        }
except:
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO(YOLO_PATH)
da3_model = DepthAnything3.from_pretrained(DA3_MODEL_ID).to(device).eval()

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_25h9)
parameters = aruco.DetectorParameters()

cap = cv2.VideoCapture(0)
if not cap.isOpened(): exit()

fingers_state = {}
frame_count = 0
last_depth_uint8 = None  # 마지막으로 계산한 깊이맵 캐싱

print("=== 최적화된 키보드 시스템 시작 (종료: q) ===")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1

    # ----------------------------------------------------
    # [1] 환경 인식 (마커) - 매 프레임 수행
    # ----------------------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    matrix = None
    warped_view = np.zeros((WARP_H, WARP_W, 3), dtype=np.uint8)

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

                # [최적화] Depth는 가끔만 계산 (Skip Frame)
                if frame_count % DEPTH_SKIP_FRAMES == 0 or last_depth_uint8 is None:
                    # 1. 워핑된 이미지로 추론
                    warped_img_da3 = cv2.warpPerspective(frame, matrix, (WARP_W, WARP_H))
                    da3_res = da3_model.inference([warped_img_da3])
                    depth_map = da3_res.depth[0]

                    # 2. 정규화 (0~255)
                    d_min, d_max = depth_map.min(), depth_map.max()
                    depth_norm = (depth_map - d_min) / (d_max - d_min)
                    temp_uint8 = (depth_norm * 255).astype(np.uint8)

                    # ★ [핵심 수정] 크기 강제 맞춤 (IndexError 방지) ★
                    # 모델 출력 크기가 WARP 설정과 다를 수 있으므로 리사이즈 필수
                    if temp_uint8.shape[:2] != (WARP_H, WARP_W):
                        temp_uint8 = cv2.resize(temp_uint8, (WARP_W, WARP_H))

                    last_depth_uint8 = temp_uint8

                aruco.drawDetectedMarkers(frame, corners, ids)
            except:
                pass

    # 키보드 그리기
    for key_name, rect in KEY_LAYOUT.items():
        rx, ry, rw, rh = rect['x'], rect['y'], rect['w'], rect['h']
        cv2.rectangle(warped_view, (rx, ry), (rx + rw, ry + rh), (0, 100, 0), 1)
        cv2.putText(warped_view, key_name, (rx + 5, ry + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # ----------------------------------------------------
    # [2] 손가락 추적 - 매 프레임 수행
    # ----------------------------------------------------
    results = yolo_model.track(frame, persist=True, verbose=False, device=device)
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
                    'depth_history': collections.deque(maxlen=HISTORY_LEN),
                    'prev_pos': None, 'prev_time': 0,
                    'state': 'hover', 'last_input': 0, 'stop_time': 0, 'hover_key': None
                }
            st = fingers_state[track_id]

            # 좌표 추출
            x1, y1, x2, y2 = map(int, box)
            fx, fy = (x1 + x2) / 2, y2 * 0.9 + y1 * 0.1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if matrix is not None and last_depth_uint8 is not None:
                pts = np.array([[[fx, fy]]], dtype=np.float32)
                trans = cv2.perspectiveTransform(pts, matrix)
                tx, ty = trans[0][0]

                if 0 <= tx < WARP_W and 0 <= ty < WARP_H:
                    # [안정화 로직] 현재 깊이값 (최근 계산된 맵 사용)
                    curr_z = int(last_depth_uint8[int(ty), int(tx)])

                    # 히스토리에 추가
                    st['depth_history'].append(curr_z)

                    # ★ [핵심] 중앙값(Median) 필터로 노이즈 제거 후 평균 사용
                    valid_history = sorted(list(st['depth_history']))
                    # 극단값(최소/최대) 제거하고 중간값들만 평균
                    if len(valid_history) >= 3:
                        avg_z = int(np.mean(valid_history[1:-1]))
                    else:
                        avg_z = int(np.mean(valid_history))

                    # 깊이 변화량 (평균 - 현재) -> 튀는 값 방어됨
                    depth_diff = avg_z - curr_z

                    # 속도 계산
                    speed = 9999.0
                    if st['prev_pos'] is not None and st['prev_time'] > 0:
                        dt = curr_time - st['prev_time']
                        if dt > 0:
                            dist = np.linalg.norm(np.array([tx, ty]) - np.array(st['prev_pos']))
                            speed = dist / dt

                    # 시각화
                    color = (0, 0, 255) if st['state'] == 'hover' else (0, 255, 0)
                    cv2.circle(warped_view, (int(tx), int(ty)), 5, color, -1)

                    # 키 히트 테스트
                    detected_key = None
                    for key_name, rect in KEY_LAYOUT.items():
                        rx, ry, rw, rh = rect['x'], rect['y'], rect['w'], rect['h']
                        if rx < tx < rx + rw and ry < ty < ry + rh:
                            detected_key = key_name
                            break

                    # 입력 로직
                    if detected_key:
                        if speed < STOP_SPEED_THRESHOLD and depth_diff > TOUCH_DEPTH_DIFF:
                            if st['state'] == 'hover' and (curr_time - st['last_input']) > 0.2:
                                print(f"👉 Touch({track_id}): {detected_key} (Diff:{depth_diff})")

                                py_key = SPECIAL_KEYS.get(detected_key, detected_key.lower())
                                if py_key: pyautogui.press(py_key)

                                st['state'] = 'touch'
                                st['last_input'] = curr_time
                                cv2.rectangle(warped_view, (0, 0), (WARP_W, WARP_H), (0, 255, 0), 5)

                    elif st['state'] == 'touch' and depth_diff < 5:
                        st['state'] = 'hover'

                    st['prev_pos'] = (tx, ty)
                    st['prev_time'] = curr_time

    expired_ids = [k for k in fingers_state.keys() if k not in current_ids]
    for k in expired_ids: del fingers_state[k]

    cv2.imshow("Optimized Keyboard", warped_view)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()