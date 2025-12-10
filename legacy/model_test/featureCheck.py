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
YOLO_PATH = r'indexFinger_best.pt'
CLASSIFIER_PATH = os.path.join(current_dir, "touch_classifier_best.pth")

COOLDOWN_TIME = 0.2
TOUCH_MIN_DURATION = 0.1

SPECIAL_KEYS = {
    "SpaceBar": "space", "Enter": "enter", "Backspace": "backspace",
    "Tab": "tab", "CapsRock": "capslock", "Shift": "shift",
    "RShift": "shiftright", "Ctrl": "ctrl", "Win": "win",
    "Alt": "alt", "up": "up", "down": "down",
    "left": "left", "right": "right", "~": "`"
}


# ==========================================
# 1. Grad-CAM 및 모델 정의
# ==========================================
class TouchClassifier(nn.Module):
    def __init__(self):
        super(TouchClassifier, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            # Block 3 (Target Layer for Grad-CAM is here: Index 8)
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


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook 등록
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        # 1. 순전파 (Forward)
        self.model.zero_grad()
        output = self.model(input_tensor)

        # 2. 역전파 (Backward) - 해당 클래스의 스코어를 미분
        score = output[0, class_idx]
        score.backward()

        # 3. Grad-CAM 계산
        gradients = self.gradients[0]  # (channels, h, w)
        activations = self.activations[0]  # (channels, h, w)

        # 채널별 가중치 계산 (Global Average Pooling)
        weights = torch.mean(gradients, dim=(1, 2))

        # 가중치를 적용하여 맵 생성
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=gradients.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU 적용
        cam = torch.relu(cam)

        # 정규화 및 이미지 변환
        cam = cam.cpu().detach().numpy()
        cam = cv2.resize(cam, (64, 64))  # 입력 이미지 크기로 복원
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-7)  # 0~1 사이로 정규화

        return cam


# ==========================================
# 2. 초기화
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 디바이스: {device}")

yolo_model = YOLO(YOLO_PATH)
touch_model = TouchClassifier().to(device)
try:
    touch_model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    touch_model.eval()  # 평가 모드지만 Grad-CAM을 위해 그라디언트는 계산함
except:
    print("❌ 모델 로드 실패")
    exit()

# Grad-CAM 설정 (마지막 Conv 레이어인 features[8] 타겟팅)
grad_cam = GradCAM(touch_model, touch_model.features[8])

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

print("=== Explainable AI Keyboard 시작 (종료: q) ===")

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    matrix = None
    curr_marker_centers = {}

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        ids = ids.flatten()
        for i, tag_id in enumerate(ids):
            curr_marker_centers[tag_id] = corners[i][0].mean(axis=0)

    # -----------------------------------------------------------
    # [1] 마커 추적 및 맵핑 (간소화된 로직)
    # -----------------------------------------------------------
    if ids is not None and len(ids) >= 4:
        corners_map = {id: corner for id, corner in zip(ids, corners)}
        if all(i in corners_map for i in [0, 1, 2, 3]):
            try:
                src_pts = np.array([
                    corners_map[0][0][1], corners_map[1][0][0],
                    corners_map[3][0][3], corners_map[2][0][2]
                ], dtype=np.float32)
                dst_pts = np.array([[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]], dtype=np.float32)
                matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                prev_matrix = matrix.copy()
                prev_paper_corners = src_pts.copy()
                prev_marker_centers = curr_marker_centers.copy()
            except:
                pass
    elif len(curr_marker_centers) >= 2 and prev_paper_corners is not None:
        common_ids = []
        pts_prev, pts_curr = [], []
        for tag_id in curr_marker_centers:
            if tag_id in prev_marker_centers:
                common_ids.append(tag_id)
                pts_prev.append(prev_marker_centers[tag_id])
                pts_curr.append(curr_marker_centers[tag_id])
        if len(common_ids) >= 2:
            M, _ = cv2.estimateAffinePartial2D(np.array(pts_prev), np.array(pts_curr))
            if M is not None:
                prev_pts_reshaped = prev_paper_corners.reshape(-1, 1, 2)
                curr_paper_corners = cv2.transform(prev_pts_reshaped, M).reshape(4, 2)
                dst_pts = np.array([[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]], dtype=np.float32)
                matrix = cv2.getPerspectiveTransform(curr_paper_corners.astype(np.float32), dst_pts)
                prev_matrix = matrix
                prev_paper_corners = curr_paper_corners
                prev_marker_centers = curr_marker_centers
    elif prev_matrix is not None:
        matrix = prev_matrix

    #if prev_paper_corners is not None:
    #    cv2.polylines(frame, [prev_paper_corners.astype(np.int32)], True, (0, 0, 255), 2)

    warped_view = np.zeros((WARP_H, WARP_W, 3), dtype=np.uint8)
    for key_name, rect in KEY_LAYOUT.items():
        rx, ry, rw, rh = rect['x'], rect['y'], rect['w'], rect['h']
        cv2.rectangle(warped_view, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 1)
        cv2.putText(warped_view, key_name, (rx + 5, ry + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # -----------------------------------------------------------
    # [2] 손가락 추적 및 Grad-CAM 시각화
    # -----------------------------------------------------------
    results = yolo_model.track(frame, persist=True, verbose=False, device=device)
    curr_time = time.time()
    current_ids = set()

    # 시각화용 이미지 리스트 (히트맵 저장)
    cam_visualizations = []

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

            is_touch_visual = False

            # [Grad-CAM 실행]
            if finger_img.size > 0:
                pil_img = Image.fromarray(cv2.cvtColor(finger_img, cv2.COLOR_BGR2RGB))
                input_tensor = transform(pil_img).unsqueeze(0).to(device)

                # 예측 및 히트맵 생성
                # 주의: Backward를 해야 하므로 no_grad()를 쓰면 안 됨
                output = touch_model(input_tensor)
                prob = torch.softmax(output, dim=1)
                pred_idx = torch.argmax(prob, dim=1).item()
                is_touch_visual = (pred_idx == 1)  # Class 1 = Touch 가정

                # 히트맵 생성 (예측된 클래스 기준)
                heatmap = grad_cam.generate_heatmap(input_tensor, pred_idx)

                # 히트맵 시각화 (원본 이미지 + 히트맵)
                heatmap = cv2.resize(heatmap, (finger_img.shape[1], finger_img.shape[0]))
                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                superimposed_img = cv2.addWeighted(finger_img, 0.6, heatmap_color, 0.4, 0)

                # 결과창에 띄울 이미지 생성 (테두리 포함)
                border_color = (0, 0, 255) if is_touch_visual else (0, 255, 0)  # Touch=Red, Hover=Green
                vis_img = cv2.copyMakeBorder(superimposed_img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=border_color)

                # 라벨 추가
                status_str = f"ID:{track_id} {'TOUCH' if is_touch_visual else 'HOVER'} ({prob[0][1]:.2f})"
                vis_img = cv2.resize(vis_img, (128, 128))  # 보기 좋게 확대
                cv2.putText(vis_img, status_str, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cam_visualizations.append(vis_img)

            # 시각화 (메인 화면)
            fx, fy = (x1 + x2) / 2, (y1 - y2) / 3 + y2
            color = (0, 255, 255) if is_touch_visual else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{'TOUCH' if is_touch_visual else 'HOVER'}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

            # 입력 로직
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

                    if detected_key and is_touch_visual:
                        if st['touch_start_time'] == 0: st['touch_start_time'] = curr_time
                        if (curr_time - st['touch_start_time']) >= TOUCH_MIN_DURATION:
                            if not st['is_touching'] and (curr_time - st['last_input'] > COOLDOWN_TIME):
                                print(f"👉 Input({track_id}): {detected_key}")
                                py_key = SPECIAL_KEYS.get(detected_key, detected_key.lower())
                                if py_key: pyautogui.press(py_key)
                                st['last_input'] = curr_time
                                st['is_touching'] = True
                                rx, ry, rw, rh = KEY_LAYOUT[detected_key]['x'], KEY_LAYOUT[detected_key]['y'], \
                                KEY_LAYOUT[detected_key]['w'], KEY_LAYOUT[detected_key]['h']
                                cv2.rectangle(warped_view, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), -1)
                    else:
                        st['touch_start_time'] = 0
                        st['is_touching'] = False

    # ID 정리
    expired_ids = [k for k in fingers_state.keys() if k not in current_ids]
    for k in expired_ids: del fingers_state[k]

    # [XAI View] Grad-CAM 결과들을 가로로 이어 붙여서 보여줌
    if cam_visualizations:
        xai_view = np.hstack(cam_visualizations)
        cv2.imshow("Explainable AI (Grad-CAM)", xai_view)
    else:
        # 손가락 없으면 빈 창
        cv2.imshow("Explainable AI (Grad-CAM)", np.zeros((128, 128, 3), dtype=np.uint8))

    cv2.imshow("Tracking Cam", frame)
    cv2.imshow("AI Keyboard", warped_view)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()