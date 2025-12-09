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
# 0. 설정 및 모델 정의
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))

WARP_W = 1200
WARP_H = 620
LAYOUT_FILE = "key_layout.json"
YOLO_PATH = r'finger_project/finger_project/train_result/weights/YUN_best.pt'
CLASSIFIER_PATH = os.path.join(current_dir, "touch_classifier_best.pth")
COOLDOWN_TIME = 0.2

# 특수 키 매핑
SPECIAL_KEYS = {
    "SpaceBar": "space", "Enter": "enter", "Backspace": "backspace",
    "Tab": "tab", "CapsRock": "capslock", "Shift": "shift",
    "RShift": "shiftright", "Ctrl": "ctrl", "Win": "win",
    "Alt": "alt", "up": "up", "down": "down",
    "left": "left", "right": "right", "~": "`"
}


# CNN 모델 클래스 (학습된 모델 구조와 100% 일치해야 함)
class TouchClassifier(nn.Module):
    def __init__(self):
        super(TouchClassifier, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # 학습 코드에 추가했던 부분
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # 학습 코드에 추가했던 부분
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # 학습 코드에 추가했던 부분
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),  # 학습 코드는 256이었습니다.
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # 256 -> 2
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

# 모델 로드
yolo_model = YOLO(YOLO_PATH)

touch_model = TouchClassifier().to(device)
try:
    touch_model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    touch_model.eval()
    print("✅ 터치 분류 모델 로드 성공")
except Exception as e:
    print(f"오류발생: {e}")
    print(f"❌ {CLASSIFIER_PATH} 파일이 없습니다. 2단계 학습을 먼저 하세요.")
    exit()

# 전처리기 (이미지 -> 텐서)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# JSON 로드
with open(LAYOUT_FILE, "r", encoding='utf-8') as f:
    raw_layout = json.load(f)
KEY_LAYOUT = {}
for k, v in raw_layout.items():
    KEY_LAYOUT[k] = {'x': v[0], 'y': v[1], 'w': v[2], 'h': v[3]}

# ArUco 설정
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_25h9)
parameters = aruco.DetectorParameters()

cap = cv2.VideoCapture(0)
if not cap.isOpened(): exit()

# 상태 변수
fingers_state = {}

print("=== AI 터치 키보드 시작 (종료: q) ===")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. 마커 탐지 & 변환 행렬
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
                aruco.drawDetectedMarkers(frame, corners, ids)
            except:
                pass

    # 키보드 그리기
    for key_name, rect in KEY_LAYOUT.items():
        rx, ry, rw, rh = rect['x'], rect['y'], rect['w'], rect['h']
        cv2.rectangle(warped_view, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 1)
        cv2.putText(warped_view, key_name, (rx + 5, ry + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # 2. YOLO 손가락 추적
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
                fingers_state[track_id] = {'last_input': 0, 'is_touching': False}
            st = fingers_state[track_id]

            # 좌표 추출
            x1, y1, x2, y2 = map(int, box)

            # [이미지 크롭] 손가락 모양 분석용
            h, w, _ = frame.shape
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(w, x2), min(h, y2)
            finger_img = frame[cy1:cy2, cx1:cx2]

            # [터치 분류]
            is_touch_visual = False
            if finger_img.size > 0:
                # PIL 변환 -> 텐서 -> GPU
                pil_img = Image.fromarray(cv2.cvtColor(finger_img, cv2.COLOR_BGR2RGB))
                input_tensor = transform(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = touch_model(input_tensor)
                    prob = torch.softmax(output, dim=1)
                    # Class 1이 Touch라고 가정 (학습 데이터 폴더 순서에 따라 다를 수 있음)
                    # 보통 알파벳 순이므로 hover=0, touch=1일 확률 높음. 확인 필요.
                    is_touch_visual = prob[0][1].item() > 0.5

                    # 좌표 변환
            fx, fy = (x1 + x2) / 2, y2 * 0.9 + y1 * 0.1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            status_text = "TOUCH" if is_touch_visual else "HOVER"
            cv2.putText(frame, status_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if is_touch_visual else (0, 255, 0), 2)

            if matrix is not None:
                pts = np.array([[[fx, fy]]], dtype=np.float32)
                trans = cv2.perspectiveTransform(pts, matrix)
                tx, ty = trans[0][0]

                if 0 <= tx < WARP_W and 0 <= ty < WARP_H:
                    cv2.circle(warped_view, (int(tx), int(ty)), 8, (0, 0, 255), -1)

                    # 키 히트 테스트
                    detected_key = None
                    for key_name, rect in KEY_LAYOUT.items():
                        rx, ry, rw, rh = rect['x'], rect['y'], rect['w'], rect['h']
                        if rx < tx < rx + rw and ry < ty < ry + rh:
                            detected_key = key_name
                            break

                    # [최종 입력 판단]
                    # 시각적으로 'Touch' 상태이고 + 쿨다운이 지났으면 입력
                    if detected_key and is_touch_visual:
                        if not st['is_touching'] and (curr_time - st['last_input'] > COOLDOWN_TIME):
                            print(f"👉 Input({track_id}): {detected_key}")

                            py_key = SPECIAL_KEYS.get(detected_key, detected_key.lower())
                            if py_key: pyautogui.press(py_key)

                            st['last_input'] = curr_time
                            st['is_touching'] = True

                            # 시각 효과
                            rx, ry, rw, rh = KEY_LAYOUT[detected_key]['x'], KEY_LAYOUT[detected_key]['y'], \
                            KEY_LAYOUT[detected_key]['w'], KEY_LAYOUT[detected_key]['h']
                            cv2.rectangle(warped_view, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), -1)

                    elif not is_touch_visual:
                        st['is_touching'] = False

    # ID 정리
    expired_ids = [k for k in fingers_state.keys() if k not in current_ids]
    for k in expired_ids: del fingers_state[k]

    cv2.imshow("Tracking Cam", frame)
    cv2.imshow("AI Keyboard", warped_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()