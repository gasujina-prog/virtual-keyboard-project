from ultralytics import YOLO
import cv2

# 학습 완료된 모델 로드
model = YOLO('finger_project/train_result/weights/best.pt')   # 경로는 yolo 커맨드 돌릴 때마다 train[x] 폴더 새로 생성됨.
                                                    # 여러번 돌렸으면 테스트하고 싶은 train 번호 경로 수정할 것.

# 카메라 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("카메라를 열 수 없습니다.")

print("Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 모델 추론
    results = model(frame, imgsz=640, conf=0.5, device='cuda:0')[0]  # conf 조절 가능 (기본 0.25~0.5)

    # 검출된 객체 그리기
    for box in results.boxes:
        cls = int(box.cls[0])       # 클래스 번호 (0=fingertip, 1=thumb)
        conf = float(box.conf[0])   # confidence score

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)  # 좌표
        label = model.names[cls] if model.names else str(cls)   # 클래스 이름

        # 색상 구분 (엄지는 노란색, 나머지는 파란색)
        color = (0, 255, 255) if cls == 1 else (255, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label} {conf:.2f}',
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Fingertip Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
