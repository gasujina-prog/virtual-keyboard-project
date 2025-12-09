import cv2
import numpy as np
import json
import threading
import time
from ultralytics import YOLO


class KeyboardDetector:
    def __init__(self):
        # 1. 설정
        self.JSON_PATH = "kblayout.json"
        # 절대경로/상대경로 본인 환경에 맞게 유지
        self.YOLO_WEIGHTS = "finger_project/finger_project/train_result/weights/best.pt"
        self.YOLO_CONF = 0.3
        self.YOLO_DEVICE = "cuda:0"

        self.WARP_W = 1200
        self.WARP_H = 620

        # 상태 변수
        self.frame_cam = None
        self.frame_warp = None
        self.lock = threading.Lock()
        self.running = False

        self.prev_homography = None
        self.prev_quad = None

        # ★ DB 저장을 위한 추가 변수들 ★
        self.input_queue = []  # 눌린 키를 임시 저장하는 통
        self.last_input_time = {}  # 키별 마지막 입력 시간 (중복 방지)
        self.COOLDOWN = 0.5  # 0.5초 쿨타임 (연타 방지)

        self.load_resources()
        self.cap = cv2.VideoCapture(0)

    def load_resources(self):
        with open(self.JSON_PATH, "r", encoding="utf-8") as f:
            raw_layout = json.load(f)

        self.KEY_LAYOUT = {}
        if "keys" in raw_layout:
            for k in raw_layout["keys"]:
                key_id = k.get("id", "unknown")
                self.KEY_LAYOUT[key_id] = {'x': k['x'], 'y': k['y'], 'w': k['w'], 'h': k['h']}
        else:
            for k, v in raw_layout.items():
                self.KEY_LAYOUT[k] = {'x': v[0], 'y': v[1], 'w': v[2], 'h': v[3]}

        self.model = YOLO(self.YOLO_WEIGHTS)
        self.aruco = cv2.aruco
        self.dictionary = self.aruco.getPredefinedDictionary(self.aruco.DICT_APRILTAG_25h9)
        self.parameters = self.aruco.DetectorParameters()

    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self.update)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret: continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)

            warped_view = np.zeros((self.WARP_H, self.WARP_W, 3), dtype=np.uint8)
            current_homography = None

            # ArUco 처리
            corners_np = []
            if ids is not None:
                ids = ids.flatten()
                corners_np = [np.array(c).reshape(-1, 2) for c in corners]
                self.aruco.drawDetectedMarkers(frame, corners, ids.reshape(-1, 1))

            num_tags = len(ids) if ids is not None else 0

            if num_tags >= 4:
                roles = self.classify_four_tags(corners_np, ids)
                src_pts = np.array([roles['bottom_right'], roles['bottom_left'], roles['top_left'], roles['top_right']],
                                   dtype=np.float32)
                dst_pts = np.array([[0, 0], [self.WARP_W, 0], [self.WARP_W, self.WARP_H], [0, self.WARP_H]],
                                   dtype=np.float32)
                H = cv2.getPerspectiveTransform(src_pts, dst_pts)
                current_homography = H
                self.prev_homography = H.copy()
                self.prev_quad = src_pts.copy()
            elif self.prev_homography is not None:
                current_homography = self.prev_homography

            if self.prev_quad is not None:
                self.draw_quad(frame, self.prev_quad)

            # 레이아웃 그리기
            for key, rect in self.KEY_LAYOUT.items():
                rx, ry, rw, rh = int(rect['x']), int(rect['y']), int(rect['w']), int(rect['h'])
                if rx <= 1 and ry <= 1:
                    rx, ry = int(rect['x'] * self.WARP_W), int(rect['y'] * self.WARP_H)
                    rw, rh = int(rect['w'] * self.WARP_W), int(rect['h'] * self.WARP_H)
                cv2.rectangle(warped_view, (rx, ry), (rx + rw, ry + rh), (0, 100, 0), 1)
                cv2.putText(warped_view, str(key), (rx + 5, ry + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # 손가락 탐지
            results = self.model(frame, imgsz=640, conf=self.YOLO_CONF, device=self.YOLO_DEVICE, verbose=False)[0]
            fingers = []
            for box in results.boxes:
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    fx, fy = (x1 + x2) / 2, y2
                    fingers.append((fx, fy))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # ★ 키 입력 로직 & DB 큐 저장 ★
            curr_time = time.time()
            if current_homography is not None and fingers:
                fingers_np_arr = np.array([fingers], dtype=np.float32).transpose(1, 0, 2)
                transformed_fingers = cv2.perspectiveTransform(fingers_np_arr, current_homography)

                for pt in transformed_fingers:
                    tx, ty = int(pt[0][0]), int(pt[0][1])

                    for key_name, rect in self.KEY_LAYOUT.items():
                        rx, ry, rw, rh = int(rect['x']), int(rect['y']), int(rect['w']), int(rect['h'])
                        if rx <= 1 and ry <= 1:
                            rx, ry = int(rect['x'] * self.WARP_W), int(rect['y'] * self.WARP_H)
                            rw, rh = int(rect['w'] * self.WARP_W), int(rect['h'] * self.WARP_H)

                        if rx < tx < rx + rw and ry < ty < ry + rh:
                            # 1. 시각 효과 (노란색)
                            cv2.rectangle(warped_view, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), -1)
                            cv2.putText(warped_view, str(key_name), (rx + 5, ry + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                        (0, 0, 0), 1)

                            # 2. 입력 처리 (쿨타임 체크)
                            last_t = self.last_input_time.get(key_name, 0)
                            if curr_time - last_t > self.COOLDOWN:
                                # [입력 성공]
                                print(f"👉 Input Detected: {key_name}")
                                self.last_input_time[key_name] = curr_time

                                # 큐에 저장 (Main이 가져갈 수 있게)
                                with self.lock:
                                    self.input_queue.append({
                                        "key": key_name,
                                        "time": curr_time
                                    })

                    cv2.circle(warped_view, (tx, ty), 8, (0, 0, 255), -1)

            with self.lock:
                _, buf_cam = cv2.imencode('.jpg', frame)
                _, buf_warp = cv2.imencode('.jpg', warped_view)
                self.frame_cam = buf_cam.tobytes()
                self.frame_warp = buf_warp.tobytes()

            time.sleep(0.01)

    def get_frames(self):
        with self.lock:
            return self.frame_cam, self.frame_warp

    # DB 저장을 위해 쌓인 데이터를 꺼내가는 함수
    def pop_inputs(self):
        with self.lock:
            if not self.input_queue:
                return []
            data = self.input_queue[:]  # 복사
            self.input_queue = []  # 비우기
            return data

    def draw_quad(self, frame, quad):
        quad_int = quad.astype(int)
        for i in range(4):
            cv2.line(frame, tuple(quad_int[i]), tuple(quad_int[(i + 1) % 4]), (0, 255, 0), 2)

    def classify_four_tags(self, corners, ids):
        # 3, 2, 1, 0 순서 유지
        ids = np.array(ids).flatten()
        centers = []
        for i, c in enumerate(corners):
            c_reshaped = c.reshape(-1, 2)
            cx, cy = c_reshaped.mean(axis=0)
            centers.append((i, cx, cy))

        centers = np.array(centers)
        sort_by_y = centers[np.argsort(centers[:, 2])]
        top_group = sort_by_y[:2]
        bottom_group = sort_by_y[2:]
        top_group = top_group[np.argsort(top_group[:, 1])]
        bottom_group = bottom_group[np.argsort(bottom_group[:, 1])]

        tl_idx = int(top_group[0, 0])
        tr_idx = int(top_group[1, 0])
        bl_idx = int(bottom_group[0, 0])
        br_idx = int(bottom_group[1, 0])

        return {
            "top_left": corners[tl_idx][3],
            "top_right": corners[tr_idx][2],
            "bottom_right": corners[br_idx][1],
            "bottom_left": corners[bl_idx][0],
        }