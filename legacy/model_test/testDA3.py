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

# 라이브러리 경로 강제 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

from depth_anything_3.api import DepthAnything3

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
WARP_W = 840
WARP_H = 560
LAYOUT_FILE = "key_layout.json"
YOLO_PATH = r'finger_project/finger_project/train_result/weights/YUN_best.pt'
DA3_MODEL_ID = "depth-anything/DA3-Small"

# ----------------------------------
# [튜닝 파라미터 - 감도 조절 핵심]
# ----------------------------------
# 1. 깊이 관련
DEPTH_HISTORY_LEN = 5  # 이동 평균을 구할 프레임 수 (클수록 안정적, 반응 느림)
TOUCH_DEPTH_DIFF = 15  # (평균 - 현재) 차이가 이보다 크면 '터치' (쑥 내려감)
RELEASE_DEPTH_DIFF = 10  # 차이가 이보다 작아지면 '땜' (다시 올라옴)

# 2. 속도 관련
STOP_SPEED_THRESHOLD = 80.0  # 이 속도(px/s)보다 빠르면 터치 인정 안 함 (이동 중 오타 방지)

# 3. 입력 쿨다운
COOLDOWN_TIME = 0.1
# ----------------------------------

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

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 실행 디바이스: {device}")

# 모델 로드
yolo_model = YOLO(YOLO_PATH)
try:
    da3_model = DepthAnything3.from_pretrained(DA3_MODEL_ID).to(device).eval()
except Exception as e:
    print(f"DA3 로드 실패: {e}")
    exit()

# ArUco 설정
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_25h9)
parameters = aruco.DetectorParameters()

cap = cv2.VideoCapture(0)
if not cap.isOpened(): exit()

# ==========================================
# 3. 상태 변수 (손가락별 데이터)
# ==========================================
# 구조: { track_id : { 'history': deque, 'prev_pos': (x,y), 'prev_time': t, 'state': 'hover'/'touch' } }
fingers_state = {}

print("=== 최종 하이브리드 키보드 시스템 시작 (종료: q) ===")

while True:
    ret, frame = cap.read()
    if not ret: break

    # ----------------------------------------------------
    # [1] 환경 인식 (마커 & Depth Map 생성)
    # ----------------------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    matrix = None
    warped_view = np.zeros((WARP_H, WARP_W, 3), dtype=np.uint8)
    depth_uint8 = None

    if ids is not None and len(ids) >= 4:
        ids = ids.flatten()
        corners_map = {id: corner for id, corner in zip(ids, corners)}

        if all(i in corners_map for i in [0, 1, 2, 3]):
            try:
                # 좌표 순서: TL(0), TR(1), BR(3), BL(2) (사용자 마커 배치 기준)
                src_pts = np.array([
                    corners_map[0][0][1], corners_map[1][0][0],
                    corners_map[3][0][3], corners_map[2][0][2]
                ], dtype=np.float32)
                dst_pts = np.array([
                    [0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]
                ], dtype=np.float32)

                matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

                # DA3용 워핑 이미지
                warped_img_da3 = cv2.warpPerspective(frame, matrix, (WARP_W, WARP_H))

                # DA3 추론 (가상 화면 전체)
                da3_res = da3_model.inference([warped_img_da3])
                depth_map = da3_res.depth[0]

                # 정규화 (0~255)
                d_min, d_max = depth_map.min(), depth_map.max()
                depth_norm = (depth_map - d_min) / (d_max - d_min)
                depth_uint8 = (depth_norm * 255).astype(np.uint8)

                # 리사이즈 (혹시 크기 다를 경우)
                if depth_uint8.shape[:2] != (WARP_H, WARP_W):
                    depth_uint8 = cv2.resize(depth_uint8, (WARP_W, WARP_H))

                aruco.drawDetectedMarkers(frame, corners, ids)
            except:
                pass

    # 가상 키보드 그리기
    for key_name, rect in KEY_LAYOUT.items():
        rx, ry, rw, rh = rect['x'], rect['y'], rect['w'], rect['h']
        cv2.rectangle(warped_view, (rx, ry), (rx + rw, ry + rh), (0, 100, 0), 1)
        cv2.putText(warped_view, key_name, (rx + 5, ry + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # ----------------------------------------------------
    # [2] 손가락 추적 및 상태 업데이트
    # ----------------------------------------------------
    # YOLO 추적 (GPU)
    results = yolo_model.track(frame, persist=True, verbose=False, device=device)
    curr_time = time.time()
    current_ids = set()

    for r in results:
        if r.boxes.id is None: continue
        boxes = r.boxes.xyxy.cpu().numpy()
        track_ids = r.boxes.id.int().cpu().numpy()

        for box, track_id in zip(boxes, track_ids):
            current_ids.add(track_id)

            # [2-1] 상태 초기화 (새로운 손가락)
            if track_id not in fingers_state:
                fingers_state[track_id] = {
                    'history': collections.deque(maxlen=DEPTH_HISTORY_LEN),  # 깊이 기록 (큐)
                    'prev_pos': None,
                    'prev_time': 0,
                    'state': 'hover',  # 현재 상태: hover(뜸) / touch(누름)
                    'last_input': 0
                }

            st = fingers_state[track_id]

            # 좌표 추출 (박스 하단)
            x1, y1, x2, y2 = map(int, box)
            fx = (x1 + x2) / 2
            fy = (y1 - y2) / 3 + y2

            # 원본 화면 표시
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # [2-2] 데이터 계산 (좌표 변환 -> 속도/깊이)
            if matrix is not None and depth_uint8 is not None:
                # 좌표 변환
                pts = np.array([[[fx, fy]]], dtype=np.float32)
                trans = cv2.perspectiveTransform(pts, matrix)
                tx, ty = trans[0][0]

                # 범위 체크
                if 0 <= tx < WARP_W and 0 <= ty < WARP_H:
                    # A. 현재 깊이값 (DA3)
                    curr_z = int(depth_uint8[int(ty), int(tx)])

                    # B. 히스토리 업데이트 (터치 중이 아닐 때만 업데이트하거나, 천천히 업데이트)
                    # 여기서는 항상 업데이트하되, 터치 판단은 '평균'과 비교함
                    st['history'].append(curr_z)
                    avg_z = int(np.mean(st['history']))

                    # C. 깊이 변화량 (평균 - 현재)
                    # 손가락이 내려가면 curr_z가 작아짐 (어두워짐) -> diff가 커짐 (+)
                    # (DA3 특성: 가까움=밝음/큼, 멈=어두움/작음)
                    # 만약 손가락이 내려갈 때 값이 커지는 모델이라면 반대로 계산해야 함
                    # 일반적으로: 바닥(어두움) < 손(밝음). 손이 바닥으로 가면 어두워짐.
                    # 즉, 내려갈 때 값 감소 -> (평균 - 현재) > 0
                    depth_diff = avg_z - curr_z

                    # D. 속도 계산
                    speed = 9999.0
                    if st['prev_pos'] is not None and st['prev_time'] > 0:
                        dt = curr_time - st['prev_time']
                        if dt > 0:
                            dist = np.linalg.norm(np.array([tx, ty]) - np.array(st['prev_pos']))
                            speed = dist / dt

                    # 시각화
                    color = (0, 0, 255) if st['state'] == 'hover' else (0, 255, 0)
                    cv2.circle(warped_view, (int(tx), int(ty)), 8, color, -1)
                    cv2.putText(warped_view, f"Diff:{depth_diff} Spd:{int(speed)}",
                                (int(tx), int(ty) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # [2-3] 키 히트 테스트
                    detected_key = None
                    for key_name, rect in KEY_LAYOUT.items():
                        rx, ry, rw, rh = rect['x'], rect['y'], rect['w'], rect['h']
                        if rx < tx < rx + rw and ry < ty < ry + rh:
                            detected_key = key_name
                            break

                    # ==================================================
                    # ★ [핵심] 상태 머신 (State Machine) 로직 ★
                    # ==================================================

                    # 1. 터치 감지 (Hover -> Touch)
                    if st['state'] == 'hover':
                        # 조건: 키 위에 있음 AND 속도가 안정됨 AND 깊이가 쑥 들어감
                        if detected_key and speed < STOP_SPEED_THRESHOLD and depth_diff > TOUCH_DEPTH_DIFF:

                            if (curr_time - st['last_input']) > COOLDOWN_TIME:
                                print(f"👉 Touch(ID:{track_id}): {detected_key} (Diff:{depth_diff})")

                                # 입력 실행
                                py_key = SPECIAL_KEYS.get(detected_key, detected_key.lower())
                                if py_key: pyautogui.press(py_key)

                                # 상태 변경 및 쿨다운
                                st['state'] = 'touch'
                                st['last_input'] = curr_time

                                # 시각 효과
                                rx, ry, rw, rh = KEY_LAYOUT[detected_key]['x'], KEY_LAYOUT[detected_key]['y'], \
                                KEY_LAYOUT[detected_key]['w'], KEY_LAYOUT[detected_key]['h']
                                cv2.rectangle(warped_view, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), -1)

                    # 2. 릴리즈 감지 (Touch -> Hover)
                    elif st['state'] == 'touch':
                        # 조건: 깊이 차이가 줄어들면 (다시 올라오면) 해제
                        if depth_diff < RELEASE_DEPTH_DIFF:
                            st['state'] = 'hover'
                            print(f"💨 Release(ID:{track_id})")

                    # 상태 업데이트
                    st['prev_pos'] = (tx, ty)
                    st['prev_time'] = curr_time

    # 사라진 ID 정리
    expired_ids = [k for k in fingers_state.keys() if k not in current_ids]
    for k in expired_ids: del fingers_state[k]

    # 화면 출력
    cv2.imshow("Tracking Cam", frame)
    cv2.imshow("Hybrid Keyboard", warped_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()