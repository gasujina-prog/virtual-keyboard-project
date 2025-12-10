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
IMG_SIZE = 64  # 저장될 이미지 크기 (CNN 입력용)
PADDING = 20  # 박스 주변 여백
CONF_THRESHOLD = 0.68  # 객체 정확도 임계값

# ★ [추가] 자동 저장 설정 ★
SAVE_INTERVAL = 0.2  # 자동 저장 간격 (초 단위, 예: 0.2초마다 저장)

# 폴더 생성
os.makedirs(f"{SAVE_DIR}/touch", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/hover", exist_ok=True)

# 모델 & 카메라 로드
model = YOLO(YOLO_PATH)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 현재 저장된 개수 확인
cnt_touch = len(os.listdir(f"{SAVE_DIR}/touch"))
cnt_hover = len(os.listdir(f"{SAVE_DIR}/hover"))

# 자동 저장 관련 변수
auto_mode = False  # True면 자동 저장 중
auto_target = None  # 'touch' 또는 'hover'
last_save_time = 0  # 마지막 저장 시간

print("=== 자동 데이터 수집기 시작 ===")
print(f"👉 설정: Conf {CONF_THRESHOLD}, Padding {PADDING}px, Interval {SAVE_INTERVAL}s")
print("------------------------------------------------")
print("👉 [수동] 't': Touch 저장 / 'h': Hover 저장")
print("👉 [자동] 'a': Auto Touch / 's': Auto Hover / 'o': 자동 멈춤")
print("👉 [종료] 'q'")
print("------------------------------------------------")

while True:
    ret, frame = cap.read()
    if not ret: break

    h_img, w_img, _ = frame.shape

    # YOLO 탐지
    results = model(frame, verbose=False)

    detected_fingers = []

    for r in results:
        for box in r.boxes:
            conf = box.conf[0].item()

            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 패딩 적용
            x1 = max(0, x1 - PADDING)
            y1 = max(0, y1 - PADDING)
            x2 = min(w_img, x2 + PADDING)
            y2 = min(h_img, y2 + PADDING)

            finger_crop = frame[y1:y2, x1:x2]

            if finger_crop.size > 0:
                finger_resized = cv2.resize(finger_crop, (IMG_SIZE, IMG_SIZE))
                detected_fingers.append((x1, y1, x2, y2, finger_resized, conf))

    # 화면 그리기
    for (x1, y1, x2, y2, img, conf) in detected_fingers:
        # 자동 저장 모드일 때 박스 색상을 다르게 표시 (빨강/파랑)
        if auto_mode:
            color = (0, 0, 255) if auto_target == 'touch' else (255, 0, 0)
        else:
            color = (0, 255, 0)  # 기본 초록색

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 미리보기 창
    if detected_fingers:
        cv2.imshow("Preview", detected_fingers[0][4])
    else:
        cv2.imshow("Preview", np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))

    # =========================================================
    # 키 입력 및 자동 저장 로직
    # =========================================================
    key = cv2.waitKey(1) & 0xFF
    curr_time = time.time()

    # 1. 모드 설정 키
    if key == ord('a'):  # Auto Touch
        auto_mode = True
        auto_target = 'touch'
        print("🟢 [AUTO] Touch 데이터 자동 수집 시작...")

    elif key == ord('s'):  # Auto Hover (키보드 배치를 고려해 a 옆 s로 설정)
        auto_mode = True
        auto_target = 'hover'
        print("🔵 [AUTO] Hover 데이터 자동 수집 시작...")

    elif key == ord('o'):  # Off (Stop)
        auto_mode = False
        auto_target = None
        print("🛑 [STOP] 자동 수집 중지")

    # 2. 저장 실행 (수동 or 자동)
    save_trigger = False
    target_label = ""

    # 수동 저장 조건
    if key == ord('t'):
        save_trigger = True
        target_label = 'touch'
    elif key == ord('h'):
        save_trigger = True
        target_label = 'hover'

    # 자동 저장 조건 (시간 간격 체크)
    if auto_mode and (curr_time - last_save_time > SAVE_INTERVAL):
        if detected_fingers:  # 감지된 게 있어야 저장
            save_trigger = True
            target_label = auto_target
            last_save_time = curr_time

    # 실제 저장 수행
    if save_trigger and detected_fingers:
        save_count = 0
        for (_, _, _, _, crop_img, _) in detected_fingers:
            timestamp = int(time.time() * 1000)

            if target_label == 'touch':
                cnt_touch += 1
                filename = f"{SAVE_DIR}/touch/touch_{cnt_touch}_{timestamp}.jpg"
            else:
                cnt_hover += 1
                filename = f"{SAVE_DIR}/hover/hover_{cnt_hover}_{timestamp}.jpg"

            cv2.imwrite(filename, crop_img)
            save_count += 1

        # 자동 모드일 때는 로그를 너무 많이 찍지 않게 화면 표시로 대체하거나 간략하게 출력
        if not auto_mode:
            print(f"📸 [{target_label.upper()}] {save_count}장 저장 완료")

    elif key == ord('q'):
        break

    # =========================================================
    # 상태 표시 UI
    # =========================================================
    cv2.putText(frame, f"Touch: {cnt_touch} | Hover: {cnt_hover}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)  # 테두리
    cv2.putText(frame, f"Touch: {cnt_touch} | Hover: {cnt_hover}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 현재 모드 표시
    if auto_mode:
        mode_text = f"AUTO MODE: {auto_target.upper()}"
        mode_color = (0, 0, 255) if auto_target == 'touch' else (255, 0, 0)
    else:
        mode_text = "MANUAL MODE"
        mode_color = (0, 255, 0)

    cv2.putText(frame, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

    cv2.imshow("Multi-Data Collector", frame)

cap.release()
cv2.destroyAllWindows()