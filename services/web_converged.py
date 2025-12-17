import cv2
import numpy as np
import json
import threading
import time
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO


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


class KeyboardDetector:
    def __init__(self):
        # [수정] services 폴더 기준 상위 폴더(루트)를 찾기 위한 경로 설정
        CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(CURRENT_FILE_DIR)

        self.JSON_PATH = os.path.join(PROJECT_ROOT, "key_layout.json")
        self.YOLO_WEIGHTS = os.path.join(PROJECT_ROOT, "modelWeight", "indexFinger_best.pt")
        self.CLASSIFIER_PATH = os.path.join(PROJECT_ROOT, "modelWeight", "touch_classifier_best.pth")

        self.WARP_W = 1200
        self.WARP_H = 620
        self.COOLDOWN_TIME = 0.1
        self.TOUCH_DWELL_TIME = 0.05
        self.AI_THRESHOLD = 0.1

        self.frame_cam = None
        self.frame_warp = None
        self.lock = threading.Lock()
        self.running = False
        self.is_active = False

        self.frame_count = 0
        self.cached_matrix = None
        self.cached_fingers_visual = []
        self.cached_marker_corners = None
        self.cached_marker_ids = None

        self.prev_matrix = None
        self.prev_paper_corners = None
        self.prev_marker_centers = {}
        self.fingers_state = {}
        self.input_queue = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 디바이스: {self.device}")

        self.load_resources()
        self.cap = cv2.VideoCapture()
        # self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def load_resources(self):
        try:
            with open(self.JSON_PATH, "r", encoding="utf-8") as f:
                raw_layout = json.load(f)
            self.KEY_LAYOUT = {}
            for k, v in raw_layout.items():
                self.KEY_LAYOUT[k] = {'x': v[0], 'y': v[1], 'w': v[2], 'h': v[3]}
        except:
            self.KEY_LAYOUT = {}

        self.aruco = cv2.aruco
        self.dictionary = self.aruco.getPredefinedDictionary(self.aruco.DICT_APRILTAG_25h9)
        self.parameters = self.aruco.DetectorParameters()

        try:
            self.yolo_model = YOLO(self.YOLO_WEIGHTS)
            self.touch_model = TouchClassifier().to(self.device)
            if os.path.exists(self.CLASSIFIER_PATH):
                self.touch_model.load_state_dict(
                    torch.load(self.CLASSIFIER_PATH, map_location=self.device, weights_only=False))
                self.touch_model.eval()
            else:
                self.touch_model = None
            self.transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
        except:
            self.yolo_model = None

    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self.update)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.cap.isOpened(): self.cap.release()

    def set_active(self, status):
        """
        카메라 토글 제어 함수
        - status=True : 카메라 재연결 (Resume)
        - status=False: 카메라 자원 해제 (Power Saving / Safe Reload)
        """
        self.is_active = status

        if not status:
            # 끄기 요청: 카메라가 켜져 있다면 전원을 끕니다.
            if self.cap.isOpened():
                self.cap.release()
                print("💤 Camera released (Power Saving Mode)")
        else:
            # 켜기 요청: 카메라가 꺼져 있다면 다시 연결합니다.
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                # 버퍼 사이즈를 1로 줄여서 지연 시간(Latency) 최소화
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print("👀 Camera restarted")

    def update(self):
        """
        백그라운드에서 계속 돌아가는 메인 루프
        """
        while self.running:
            # 1. 비활성 상태(OFF)면 아무것도 안 하고 대기 (CPU 휴식)
            if not self.is_active:
                time.sleep(0.1)
                continue

            # 2. 모델 로딩 전이면 대기
            if not self.yolo_model:
                time.sleep(1)
                continue

            # 3. 카메라 프레임 읽기 (꺼져있으면 ret=False)
            if self.cap.isOpened():
                ret, frame = self.cap.read()
            else:
                ret = False

            if not ret:
                # 카메라가 끊겼거나 다시 켜지는 중이면 잠시 대기
                time.sleep(0.1)
                continue

            self.frame_count += 1
            run_ai = (self.frame_count % 2 == 0)

            if run_ai:
                self.cached_fingers_visual = []
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = self.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
                self.cached_marker_corners = corners
                self.cached_marker_ids = ids

                matrix = None
                curr_marker_centers = {}
                if ids is not None:
                    ids_flat = ids.flatten()
                    for i, tag_id in enumerate(ids_flat):
                        curr_marker_centers[tag_id] = corners[i][0].mean(axis=0)

                if ids is not None and len(ids) >= 4:
                    corners_map = {id: corner for id, corner in zip(ids_flat, corners)}
                    if all(i in corners_map for i in [0, 1, 2, 3]):
                        try:
                            src_pts = np.array([corners_map[0][0][1], corners_map[1][0][0], corners_map[3][0][3],
                                                corners_map[2][0][2]], dtype=np.float32)
                            dst_pts = np.array([[0, 0], [self.WARP_W, 0], [self.WARP_W, self.WARP_H], [0, self.WARP_H]],
                                               dtype=np.float32)
                            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                            self.prev_matrix = matrix.copy()
                            self.prev_paper_corners = src_pts.copy()
                            self.prev_marker_centers = curr_marker_centers.copy()
                        except:
                            pass
                elif len(curr_marker_centers) >= 2 and self.prev_paper_corners is not None:
                    if self.prev_matrix is not None: matrix = self.prev_matrix
                elif self.prev_matrix is not None:
                    matrix = self.prev_matrix

                self.cached_matrix = matrix

                try:
                    results = self.yolo_model.track(frame, persist=True, verbose=False, device=self.device)
                    curr_time = time.time()
                    current_ids = set()

                    for r in results:
                        if r.boxes.id is None: continue
                        boxes = r.boxes.xyxy.cpu().numpy()
                        track_ids = r.boxes.id.int().cpu().numpy()

                        for box, track_id in zip(boxes, track_ids):
                            current_ids.add(track_id)
                            if track_id not in self.fingers_state:
                                self.fingers_state[track_id] = {'last_input': 0, 'is_touching': False,
                                                                'touch_start_time': 0, 'hover_key': None,
                                                                'hover_start_time': 0}
                            st = self.fingers_state[track_id]
                            x1, y1, x2, y2 = map(int, box)
                            y1, y2 = max(0, y1), min(frame.shape[0], y2)
                            x1, x2 = max(0, x1), min(frame.shape[1], x2)

                            finger_img = frame[y1:y2, x1:x2]
                            touch_score = 0.0
                            if finger_img.size > 0 and self.touch_model:
                                pil_img = Image.fromarray(cv2.cvtColor(finger_img, cv2.COLOR_BGR2RGB))
                                input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                                with torch.no_grad():
                                    output = self.touch_model(input_tensor)
                                    touch_score = torch.softmax(output, dim=1)[0][1].item()

                            detected_key = None
                            if matrix is not None:
                                fx, fy = (x1 + x2) / 2, (y1 - y2) / 3 + y2
                                pts = np.array([[[fx, fy]]], dtype=np.float32)
                                trans = cv2.perspectiveTransform(pts, matrix)
                                tx, ty = int(trans[0][0][0]), int(trans[0][0][1])

                                if 0 <= tx < self.WARP_W and 0 <= ty < self.WARP_H:
                                    for key_name, rect in self.KEY_LAYOUT.items():
                                        rx, ry, rw, rh = rect['x'], rect['y'], rect['w'], rect['h']
                                        if rx < tx < rx + rw and ry < ty < ry + rh:
                                            detected_key = key_name
                                            break

                            if detected_key != st['hover_key']:
                                st['hover_key'] = detected_key
                                st['hover_start_time'] = curr_time

                            trigger = False
                            if detected_key:
                                if touch_score > self.AI_THRESHOLD:
                                    trigger = True
                                elif (curr_time - st['hover_start_time']) > self.TOUCH_DWELL_TIME:
                                    trigger = True

                                if trigger:
                                    if not st['is_touching'] and (curr_time - st['last_input'] > self.COOLDOWN_TIME):
                                        print(f"👉 Input({track_id}): {detected_key}")
                                        with self.lock:
                                            if len(self.input_queue) > 100:
                                                self.input_queue = []
                                            self.input_queue.append(
                                            {"key": detected_key, "time": curr_time})
                                        st['last_input'] = curr_time
                                        st['is_touching'] = True
                            else:
                                st['is_touching'] = False
                                st['hover_start_time'] = 0

                            self.cached_fingers_visual.append({
                                "box": (x1, y1, x2, y2),
                                "text": f"{track_id}:{'HIT' if st['is_touching'] else 'HOV'}",
                                "color": (0, 255, 255) if st['is_touching'] else (0, 255, 0),
                                "key_pos": (tx, ty) if matrix is not None and 'tx' in locals() else None,
                                "detected_key": detected_key
                            })
                    expired = [k for k in self.fingers_state if k not in current_ids]
                    for k in expired: del self.fingers_state[k]
                except:
                    pass

            if self.cached_marker_ids is not None:
                self.aruco.drawDetectedMarkers(frame, self.cached_marker_corners, self.cached_marker_ids)

            warped_view = np.zeros((self.WARP_H, self.WARP_W, 3), dtype=np.uint8)
            for k, r in self.KEY_LAYOUT.items():
                cv2.rectangle(warped_view, (r['x'], r['y']), (r['x'] + r['w'], r['y'] + r['h']), (0, 100, 0), 1)
                cv2.putText(warped_view, k, (r['x'] + 5, r['y'] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200),
                            1)

            for info in self.cached_fingers_visual:
                x1, y1, x2, y2 = info['box']
                cv2.rectangle(frame, (x1, y1), (x2, y2), info['color'], 2)
                cv2.putText(frame, info['text'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, info['color'], 2)
                if info['key_pos']:
                    tx, ty = info['key_pos']
                    if 0 <= tx < self.WARP_W and 0 <= ty < self.WARP_H:
                        cv2.circle(warped_view, (tx, ty), 8, (0, 0, 255), -1)
                        if info['detected_key']:
                            r = self.KEY_LAYOUT[info['detected_key']]
                            c = (0, 0, 255) if info['color'] == (0, 255, 255) else (0, 255, 255)
                            cv2.rectangle(warped_view, (r['x'], r['y']), (r['x'] + r['w'], r['y'] + r['h']), c,
                                          -1 if c == (0, 0, 255) else 2)

            with self.lock:
                _, buf_cam = cv2.imencode('.jpg', frame)
                _, buf_warp = cv2.imencode('.jpg', warped_view)
                self.frame_cam = buf_cam.tobytes()
                self.frame_warp = buf_warp.tobytes()
            time.sleep(0.005)

    def get_frames(self):
        with self.lock: return self.frame_cam, self.frame_warp

    def pop_inputs(self):
        with self.lock:
            if not self.input_queue: return []
            data = self.input_queue[:]
            self.input_queue = []
            return data