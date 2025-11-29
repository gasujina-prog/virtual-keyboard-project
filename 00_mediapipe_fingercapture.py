import cv2
import mediapipe as mp
import time
import os

# ==========================================
# ★ 저장 모드 설정 (여기서 True/False 변경) ★
# True: 영상 저장 모드 (자동 저장)
# False: 카메라 송출 모드 (저장 안 함)
# ==========================================
SAVE_MODE = False

# 저장 주기
save_interval_sec = 1
last_save_time = 0

# 저장 폴더 구성
image_dir = 'fingercapture/image'
label_dir = 'fingercapture/label'
result_dir = 'fingercapture/labeledimage'
os.makedirs(image_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 카메라 연결
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("카메라를 열 수 없습니다. 연결 상태 확인하세요.")

print("# ==========================================\n"
      "t키: 영상 저장모드 변경\ns키: 수동 저장\nq: 종료\n"
      "# ==========================================")

while True:
    ret, orig_frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    h, w, _ = orig_frame.shape

    vis_frame = orig_frame.copy()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 4번 엄지, 나머지 손가락
            for idx in [4,8,12,16,20]:
                lm = hand_landmarks.landmark[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)

                ################################################
                # bbox margin 설정
                # 카메라 높이에 따라서 손가락 사이즈 달라짐,
                # 손가락 (손톱 포함?) 끝 마디가 반이상 들어오도록 조정
                ################################################
                bbox_size = 40  # 박스 사이즈 px 단위
                x1 = max(cx - bbox_size // 2, 0)
                y1 = max(cy - bbox_size // 2, 0)
                x2 = min(cx + bbox_size // 2, w)
                y2 = min(cy + bbox_size // 2, h)

                # 시각화
                if idx in [8,12,16,20]:
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                    cv2.putText(vis_frame, "tip", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                else:
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                    cv2.putText(vis_frame, "thumb", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255  ), 1)


    # 화면으로 보이는 것
    cv2.imshow("Camera", vis_frame)

    current_time = time.time()
    key = cv2.waitKey(1) & 0xFF

    # 't' 키를 누르면 실행 도중 모드 변경 가능
    if key == ord('t'):
        SAVE_MODE = not SAVE_MODE
        print(f"모드 변경됨: {'[저장 ON]' if SAVE_MODE else '[저장 OFF]'}")

    # ====================================================
    # 저장 로직
    # 조건 1: SAVE_MODE가 True이고, 지정된 시간이 지났을 때 (자동 저장)
    # 조건 2: 's' 키를 눌렀을 때 (수동 저장 - 모드 상관없이 동작)
    # ====================================================
    should_save = False

    if SAVE_MODE and (current_time - last_save_time > save_interval_sec):
        should_save = True

    if key == ord('s'):
        should_save = True

    if should_save:
        #################################
        # 저장되는 파일 이름, 취향껏 수정
        #################################
        filename = f"jun_{int(current_time)}.jpg"
        img_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt"))
        result_path = os.path.join(result_dir, filename)

        # YOLO 포맷 라벨 생성
        with open(label_path, "w") as f:
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for idx in [4, 8, 12, 16, 20]:
                        lm = hand_landmarks.landmark[idx]
                        cx, cy = lm.x, lm.y  # 이미 정규화 되어 있음 (0~1)

                        # 정규화 박스 크기 (40px 상대 비율)
                        bbox_w = bbox_size / w
                        bbox_h = bbox_size / h

                        # YOLO format: class cx cy w h
                        # 라벨링 저장
                        if idx in [8, 12, 16, 20]:
                            f.write(f"0 {cx:.6f} {cy:.6f} {bbox_w:.6f} {bbox_h:.6f}\n") # 나머지 손가락 0번
                        else:
                            f.write(f"1 {cx:.6f} {cy:.6f} {bbox_w:.6f} {bbox_h:.6f}\n") # 엄지 1번

        # 이미지 저장
        cv2.imwrite(result_path, vis_frame)
        cv2.imwrite(img_path, orig_frame)
        print(f"[+] 저장 완료 → {img_path}")
        print(f"    라벨   → {label_path}")
        last_save_time = current_time

    # 종료는 q 키
    if key == ord('q'):
        print("종료합니다.")
        break

cap.release()
cv2.destroyAllWindows()
