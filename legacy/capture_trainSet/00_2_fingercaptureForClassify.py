import cv2
import os
import numpy as np
from ultralytics import YOLO
import time

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
YOLO_PATH = r'finger_project/finger_project/train_result/weights/YUN_best.pt'
SAVE_DIR = "touch_dataset"
IMG_SIZE = 64
PADDING = 20
CONF_THRESHOLD = 0.5

# 폴더 생성
os.makedirs(f"{SAVE_DIR}/touch", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/hover", exist_ok=True)

# 모델 & 카메라
model = YOLO(YOLO_PATH)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라 오류")
    exit()

# 변수 초기화
cnt_touch = len(os.listdir(f"{SAVE_DIR}/touch"))
cnt_hover = len(os.listdir(f"{SAVE_DIR}/hover"))
detected_fingers = []  # 현재 프레임의 감지된 손가락 정보
target_id = None  # ★ 내가 선택한 손가락 ID (None이면 전체)


# ==========================================
# 2. 마우스 콜백 함수 (클릭해서 타겟 지정)
# ==========================================
def mouse_callback(event, x, y, flags, param):
    global target_id, detected_fingers

    # 왼쪽 클릭: 해당 위치의 손가락을 타겟으로 설정
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_something = False
        for (x1, y1, x2, y2, _, _, track_id) in detected_fingers:
            if x1 <= x <= x2 and y1 <= y <= y2:
                target_id = track_id
                print(f"👉 타겟 설정됨: ID {target_id}")
                clicked_something = True
                break

        # 빈 공간 클릭 시 타겟 해제 (전체 모드)
        if not clicked_something:
            target_id = None
            print("👉 타겟 해제 (전체 저장 모드)")


# 윈도우 생성 및 콜백 연결
cv2.namedWindow("Multi-Data Collector")
cv2.setMouseCallback("Multi-Data Collector", mouse_callback)

print("=== 타겟 트래킹 수집기 시작 ===")
print("👉 마우스 왼쪽 클릭: 저장할 손가락 선택 (빨간색)")
print("👉 빈 공간 클릭: 선택 해제 (초록색 = 전체 저장)")
print("👉 't': Touch 저장 / 'h': Hover 저장")

while True:
    ret, frame = cap.read()
    if not ret: break
    h_img, w_img, _ = frame.shape

    # 1. YOLO 추적 모드 (persist=True 필수)
    # 추적 모드를 써야 ID가 유지됩니다.
    results = model.track(frame, persist=True, verbose=False)

    detected_fingers = []  # 초기화

    for r in results:
        # 감지된 게 없거나 ID가 아직 부여 안 된 경우 패스
        if r.boxes.id is None: continue

        boxes = r.boxes.xyxy.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()
        track_ids = r.boxes.id.int().cpu().numpy()

        for box, conf, track_id in zip(boxes, confidences, track_ids):
            if conf < CONF_THRESHOLD: continue

            x1, y1, x2, y2 = map(int, box)

            # 패딩 적용 및 크롭
            px1 = max(0, x1 - PADDING)
            py1 = max(0, y1 - PADDING)
            px2 = min(w_img, x2 + PADDING)
            py2 = min(h_img, y2 + PADDING)

            finger_crop = frame[py1:py2, px1:px2]

            if finger_crop.size > 0:
                finger_resized = cv2.resize(finger_crop, (IMG_SIZE, IMG_SIZE))
                # 리스트에 ID 정보까지 함께 저장
                detected_fingers.append((x1, y1, x2, y2, finger_resized, conf, track_id))

    # 2. 화면 그리기
    target_found = False

    for (x1, y1, x2, y2, img, conf, track_id) in detected_fingers:
        # 타겟인지 확인
        is_target = (track_id == target_id)
        if is_target: target_found = True

        # 색상: 타겟이면 빨강, 아니면 초록
        color = (0, 0, 255) if is_target else (0, 255, 0)
        thickness = 3 if is_target else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # 텍스트: ID와 정확도
        label = f"ID:{track_id} ({conf:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 미리보기 창 (타겟이 있으면 타겟만, 없으면 첫 번째 것)
        if is_target:
            cv2.imshow("Preview (Target)", img)
        elif target_id is None and detected_fingers:
            cv2.imshow("Preview (Target)", detected_fingers[0][4])

    # 타겟을 잃어버렸을 때 (화면 밖으로 나감 등)
    if target_id is not None and not target_found:
        cv2.putText(frame, f"Lost Target ID:{target_id}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 3. 저장 로직
    key = cv2.waitKey(1) & 0xFF

    if key == ord('t') or key == ord('h'):
        label = "touch" if key == ord('t') else "hover"
        saved_count = 0

        for (_, _, _, _, crop_img, _, track_id) in detected_fingers:
            # ★ 핵심: 타겟이 설정되어 있으면, ID가 일치하는 것만 저장
            if target_id is not None and track_id != target_id:
                continue

            timestamp = int(time.time() * 1000)
            # 파일명에 ID도 포함 (나중에 구분하기 좋음)
            if key == ord('t'):
                cnt_touch += 1
                filename = f"{SAVE_DIR}/{label}/{label}_{cnt_touch}_ID{track_id}_{timestamp}.jpg"
            else:
                cnt_hover += 1
                filename = f"{SAVE_DIR}/{label}/{label}_{cnt_hover}_ID{track_id}_{timestamp}.jpg"

            cv2.imwrite(filename, crop_img)
            saved_count += 1

        print(f"📸 [{label.upper()}] {saved_count}장 저장 완료 (Target: {target_id if target_id else 'ALL'})")

    elif key == ord('q'):
        break

    # UI 표시
    cv2.putText(frame, f"Touch: {cnt_touch} | Hover: {cnt_hover}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    status_msg = f"TARGET: ID {target_id}" if target_id is not None else "TARGET: ALL"
    cv2.putText(frame, status_msg, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if target_id else (0, 255, 0), 2)

    cv2.imshow("Multi-Data Collector", frame)

cap.release()
cv2.destroyAllWindows()