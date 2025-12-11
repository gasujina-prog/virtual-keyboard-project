import cv2
import cv2.aruco as aruco
import numpy as np
import json
import time
import pyautogui
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import os

# ==========================================
# 0. 설정
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))

WARP_W = 1200
WARP_H = 620
LAYOUT_FILE = "key_layout.json"
YOLO_PATH = r'modelWeight\indexFinger_best.pt'  # 경로 확인!
CLASSIFIER_PATH = os.path.join(current_dir, r"modelWeight\touch_classifier_best.pth")

COOLDOWN_TIME = 0.2
TOUCH_MIN_DURATION = 0.1

SPECIAL_KEYS = {
    "SpaceBar": "space", "Enter": "enter", "Backspace": "backspace",
    "Tab": "tab", "CapsRock": "capslock", "Shift": "shift",
    "RShift": "shiftright", "Ctrl": "ctrl", "Win": "win",
    "Alt": "alt", "up": "up", "down": "down",
    "left": "left", "right": "right", "~": "`"
}


# CNN 모델 (학습 코드와 동일 구조)
class TouchClassifier(nn.Module):
    def __init__(self):
        super(TouchClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ==========================================
# 1. 초기화
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 디바이스: {device}")

yolo_model = YOLO(YOLO_PATH)
touch_model = TouchClassifier().to(device)
try:
    touch_model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    touch_model.eval()
except:
    print("❌ 모델 로드 실패")
    exit()

transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

try:
    with open(LAYOUT_FILE, "r", encoding='utf-8') as f:
        raw_layout = json.load(f)
    KEY_LAYOUT = {}
    for k, v in raw_layout.items():
        KEY_LAYOUT[k] = {'x': v[0], 'y': v[1], 'w': v[2], 'h': v[3]}
except:
    exit()

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_25h9)
parameters = aruco.DetectorParameters()

cap = cv2.VideoCapture(0)
if not cap.isOpened(): exit()

# 상태 변수
fingers_state = {}
prev_matrix = None
prev_paper_corners = None
prev_marker_centers = {}

print("=== AI 터치 키보드 (시각화 포함) 시작 ===")

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    matrix = None
    curr_marker_centers = {}

    # [시각화 1] 감지된 마커 테두리 및 ID 표시
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)  # 기본 제공 함수 사용
        ids = ids.flatten()
        for i, tag_id in enumerate(ids):
            curr_marker_centers[tag_id] = corners[i][0].mean(axis=0)

    # -----------------------------------------------------------
    # [1] 마커 추적 및 맵핑 계산
    # -----------------------------------------------------------
    # A. 4개 모두 감지 (기준점 갱신)
    if ids is not None and len(ids) >= 4:
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

                prev_matrix = matrix.copy()
                prev_paper_corners = src_pts.copy()
                prev_marker_centers = curr_marker_centers.copy()
            except:
                pass

    # B. 2~3개 감지 (이전 프레임 기반 추적)
    elif len(curr_marker_centers) >= 2 and prev_paper_corners is not None:
        common_ids = []
        pts_prev, pts_curr = [], []

        for tag_id in curr_marker_centers:
            if tag_id in prev_marker_centers:
                common_ids.append(tag_id)
                pts_prev.append(prev_marker_centers[tag_id])
                pts_curr.append(curr_marker_centers[tag_id])

        if len(common_ids) >= 2:
            pts_prev = np.array(pts_prev).reshape(-1, 1, 2)
            pts_curr = np.array(pts_curr).reshape(-1, 1, 2)
            M, _ = cv2.estimateAffinePartial2D(pts_prev, pts_curr)

            if M is not None:
                prev_pts_reshaped = prev_paper_corners.reshape(-1, 1, 2)
                curr_paper_corners = cv2.transform(prev_pts_reshaped, M).reshape(4, 2)

                dst_pts = np.array([
                    [0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]
                ], dtype=np.float32)

                matrix = cv2.getPerspectiveTransform(curr_paper_corners.astype(np.float32), dst_pts)

                prev_matrix = matrix
                prev_paper_corners = curr_paper_corners
                prev_marker_centers = curr_marker_centers

    # C. 놓침 (유지)
    elif prev_matrix is not None:
        matrix = prev_matrix

    # [시각화 2] 맵핑된 키보드 영역(종이) 테두리 그리기
    if prev_paper_corners is not None:
        # 정수형으로 변환하여 폴리곤 그리기
        paper_poly = prev_paper_corners.astype(np.int32).reshape((-1, 1, 2))
        #cv2.polylines(frame, [paper_poly], True, (0, 0, 255), 2)  # 빨간색 테두리

    # 가상 화면 그리기
    warped_view = np.zeros((WARP_H, WARP_W, 3), dtype=np.uint8)
    for key_name, rect in KEY_LAYOUT.items():
        rx, ry, rw, rh = rect['x'], rect['y'], rect['w'], rect['h']
        cv2.rectangle(warped_view, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 1)
        cv2.putText(warped_view, key_name, (rx + 5, ry + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # -----------------------------------------------------------
    # [2] 손가락 추적
    # -----------------------------------------------------------
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
                fingers_state[track_id] = {'last_input': 0, 'is_touching': False, 'touch_start_time': 0}
            st = fingers_state[track_id]

            # 좌표 & 크롭
            x1, y1, x2, y2 = map(int, box)
            h, w, _ = frame.shape
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(w, x2), min(h, y2)
            finger_img = frame[cy1:cy2, cx1:cx2]

            # 터치 분류
            is_touch_visual = False
            if finger_img.size > 0:
                pil_img = Image.fromarray(cv2.cvtColor(finger_img, cv2.COLOR_BGR2RGB))
                input_tensor = transform(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = touch_model(input_tensor)
                    prob = torch.softmax(output, dim=1)
                    is_touch_visual = prob[0][1].item() > 0.5

            # 시각화 (손가락)
            fx, fy = (x1 + x2) / 2, (y1 - y2) / 3 + y2
            color = (0, 255, 255) if is_touch_visual else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            status_text = "TOUCH" if is_touch_visual else "HOVER"
            cv2.putText(frame, status_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 좌표 변환 & 입력
            if matrix is not None:
                pts = np.array([[[fx, fy]]], dtype=np.float32)
                trans = cv2.perspectiveTransform(pts, matrix)
                tx, ty = trans[0][0]

                if 0 <= tx < WARP_W and 0 <= ty < WARP_H:
                    cv2.circle(warped_view, (int(tx), int(ty)), 8, (0, 0, 255), -1)

                    detected_key = None
                    for key_name, rect in KEY_LAYOUT.items():
                        rx, ry, rw, rh = rect['x'], rect['y'], rect['w'], rect['h']
                        if rx < tx < rx + rw and ry < ty < ry + rh:
                            detected_key = key_name
                            break

                    # 터치 지속 시간 체크
                    if detected_key and is_touch_visual:
                        if st['touch_start_time'] == 0:
                            st['touch_start_time'] = curr_time

                        duration = curr_time - st['touch_start_time']

                        if duration >= TOUCH_MIN_DURATION:
                            if not st['is_touching'] and (curr_time - st['last_input'] > COOLDOWN_TIME):
                                print(f"👉 Input({track_id}): {detected_key}")

                                py_key = SPECIAL_KEYS.get(detected_key, detected_key.lower())
                                if py_key: pyautogui.press(py_key)

                                st['last_input'] = curr_time
                                st['is_touching'] = True

                                # 입력 피드백
                                rx, ry, rw, rh = KEY_LAYOUT[detected_key]['x'], KEY_LAYOUT[detected_key]['y'], \
                                KEY_LAYOUT[detected_key]['w'], KEY_LAYOUT[detected_key]['h']
                                cv2.rectangle(warped_view, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), -1)
                    else:
                        st['touch_start_time'] = 0
                        st['is_touching'] = False

    # ID 정리
    expired_ids = [k for k in fingers_state.keys() if k not in current_ids]
    for k in expired_ids: del fingers_state[k]

    cv2.imshow("Tracking Cam", frame)
    cv2.imshow("AI Keyboard", warped_view)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
