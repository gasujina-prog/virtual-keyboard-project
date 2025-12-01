import cv2
import time
import torch
from ultralytics import YOLO
import math

# === 설정값 ===
MODEL_PATH = "finger_project/train_result/weights/best.pt"  # A모델 경로
CAM_INDEX = 0

CONF_THRES = 0.5
ALPHA_SMOOTH = 0.7
PRESS_DELTA = 0.05
RELEASE_DELTA = 0.05
MIN_SPEED = 0.001
MATCH_DIST_PX = 50          # 프레임 간 같은 손가락으로 볼 최대 거리
FINGER_TIMEOUT = 0.8        # 이 시간(초) 동안 안 보이면 그 손가락 삭제

# 사용할 클래스 (0: fingertip, 1: thumb)
VALID_CLASSES = {0, 1}


class FingerState:
    def __init__(self, cx, cy, t, h):
        y_norm = cy / h
        self.cx = cx
        self.cy = cy
        self.smooth_y = y_norm
        self.baseline_y = y_norm  # 처음엔 현재 y를 기준으로
        self.last_y = y_norm
        self.last_time = t
        self.state = "UP"
        self.press_count = 0
        self.release_count = 0
        self.last_seen = t

    def update(self, cx, cy, t, h):
        """새 관측값으로 상태 업데이트"""
        y_norm = cy / h
        self.cx = cx
        self.cy = cy
        self.last_seen = t

        # EMA 스무딩
        self.smooth_y = ALPHA_SMOOTH * y_norm + (1 - ALPHA_SMOOTH) * self.smooth_y

        # 속도 계산
        if self.last_time is None:
            vy = 0.0
        else:
            dt = t - self.last_time
            vy = (self.smooth_y - self.last_y) / dt if dt > 0 else 0.0

        self.last_time = t
        self.last_y = self.smooth_y

        # UP 상태일 때 baseline 업데이트
        if self.state == "UP":
            self.baseline_y = 0.9 * self.baseline_y + 0.1 * self.smooth_y

        # 상태 머신
        events = []  # ["PRESS", "RELEASE"] 중 발생한 것들

        if self.baseline_y is not None:
            # PRESS
            if self.state == "UP":
                if (self.smooth_y - self.baseline_y > PRESS_DELTA) and (vy > MIN_SPEED):
                    self.state = "DOWN"
                    self.press_count += 1
                    events.append("PRESS")

            # RELEASE
            elif self.state == "DOWN":
                if (self.smooth_y < self.baseline_y + RELEASE_DELTA) and (vy < 0):
                    self.state = "UP"
                    self.release_count += 1
                    events.append("RELEASE")

        return events


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    model = YOLO(MODEL_PATH)
    model.to(device)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise IOError("카메라를 열 수 없습니다.")

    # 손가락 상태 관리용: {finger_id: FingerState}
    fingers = {}
    next_finger_id = 0

    print("q 키를 누르면 종료합니다.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        now = time.time()

        # YOLO 추론
        results = model(frame, imgsz=640, conf=CONF_THRES, device=device)
        result = results[0]

        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < CONF_THRES or cls not in VALID_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                detections.append((cx, cy, cls, conf))

        # ----- 기존 fingers와 매칭 -----
        # 이번 프레임에서 업데이트된 finger id 기록
        updated_ids = set()

        for cx, cy, cls, conf in detections:
            # 가장 가까운 기존 finger 찾기
            best_id = None
            best_dist = 1e9
            for fid, f in fingers.items():
                dist = math.dist((cx, cy), (f.cx, f.cy))
                if dist < best_dist:
                    best_dist = dist
                    best_id = fid

            if best_id is not None and best_dist < MATCH_DIST_PX:
                # 기존 finger 업데이트
                f = fingers[best_id]
                events = f.update(cx, cy, now, h)
                updated_ids.add(best_id)

                # 시각화
                color = (0, 255, 255) if f.state == "UP" else (0, 0, 255)
                cv2.circle(frame, (int(f.cx), int(f.cy)), 8, color, -1)
                cv2.putText(frame, f"id{best_id}:{f.state}",
                            (int(f.cx) + 10, int(f.cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if f.baseline_y is not None:
                    y_base_px = int(f.baseline_y * h)
                    #cv2.line(frame, (0, y_base_px), (w, y_base_px), (0, 255, 0), 1)

                for ev in events:
                    print(f"[{ev}] finger {best_id}  y={f.smooth_y:.3f}")

            else:
                # 새 finger 등록
                f = FingerState(cx, cy, now, h)
                fid = next_finger_id
                next_finger_id += 1
                fingers[fid] = f
                updated_ids.add(fid)

                # 첫 프레임 시각화
                cv2.circle(frame, (int(f.cx), int(f.cy)), 8, (255, 255, 0), -1)
                cv2.putText(frame, f"id{fid}:NEW",
                            (int(f.cx) + 10, int(f.cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # ----- 오래 안 보인 손가락 제거 -----
        remove_ids = []
        for fid, f in fingers.items():
            if fid not in updated_ids:
                # 이번 프레임에 안 보인 경우: last_seen 기준으로 timeout 체크
                if now - f.last_seen > FINGER_TIMEOUT:
                    remove_ids.append(fid)

        for fid in remove_ids:
            # print(f"remove finger {fid}")
            del fingers[fid]

        cv2.imshow("Multi-finger press detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
