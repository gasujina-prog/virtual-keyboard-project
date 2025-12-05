import cv2
import torch
import numpy as np
import time
import sys
import os

# 라이브러리 경로 강제 추가 (설치 문제 대비)
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

from depth_anything_3.api import DepthAnything3

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
# Hugging Face 모델 ID (대소문자 정확해야 함)
# 404 에러가 나면 로컬 파일 경로로 바꾸세요 (예: "depth_anything_v3_vits.pth")
MODEL_ID = "depth-anything/DA3-SMALL"

# 입력 해상도 조절 (3GB VRAM 보호용)
# 518은 DINOv2의 권장 해상도 (14의 배수)
INPUT_SIZE = 630

# ==========================================
# 2. 모델 로드
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 디바이스: {device}")

try:
    # from_pretrained 사용
    model = DepthAnything3.from_pretrained(MODEL_ID)
except Exception as e:
    print(f"⚠️ 모델 로드 실패: {e}")
    print("👉 로컬 파일을 사용하려면 from_pretrained 대신 직접 로드 코드를 사용해야 합니다.")
    exit()

model = model.to(device)
model.eval()  # 평가 모드

# ==========================================
# 3. 웹캠 실행 및 추론 루프
# ==========================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다.")
    exit()

print("🎥 Depth Anything V3 웹캠 시작... (종료: 'q')")

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. 전처리: OpenCV(BGR) -> 모델 입력용 리사이즈
    # 원본 비율 유지하며 리사이즈하면 좋지만, 속도를 위해 강제 리사이즈 사용
    # (DA3는 다양한 비율을 처리할 수 있지만, 3GB VRAM에서는 크기 제한이 중요)
    # frame_resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))

    # 2. 추론 (Inference)
    # DA3 API는 BGR 이미지를 받아서 내부적으로 처리해줍니다.
    # inference()는 리스트 입력을 기대하므로 [frame]으로 감싸줍니다.
    try:
        # raw_prediction은 배치 리스트 형태로 반환됨
        prediction = model.inference([frame])

        # 3. 결과 추출 (깊이 맵)
        # prediction.depth shape: [N, H, W] -> [0]번 가져오기
        depth = prediction.depth[0]

    except Exception as e:
        print(f"추론 에러: {e}")
        break

    # 4. 시각화 (Normalization & ColorMap)
    # 깊이 값(float)을 0~255(uint8) 이미지로 변환
    depth_min = depth.min()
    depth_max = depth.max()

    # 정규화 (0~1) -> (0~255)
    depth_norm = (depth - depth_min) / (depth_max - depth_min)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)

    # 컬러맵 적용 (Inferno, Magma, Jet 등 추천)
    depth_vis = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

    # 5. 화면 출력 (원본 + 깊이맵 나란히 보기)
    # 원본 프레임 크기에 맞게 깊이맵 리사이즈 (필요시)
    if depth_vis.shape[:2] != frame.shape[:2]:
        depth_vis = cv2.resize(depth_vis, (frame.shape[1], frame.shape[0]))

    combined_view = np.hstack((frame, depth_vis))

    # FPS 표시
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time
    cv2.putText(combined_view, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Depth Anything V3 - Webcam", combined_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()