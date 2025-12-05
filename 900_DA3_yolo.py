import cv2
import cv2.aruco as aruco
import numpy as np
import time
import torch
import sys
import os
from ultralytics import YOLO

# 라이브러리 경로 강제 추가 (설치 문제 대비)
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

from depth_anything_3.api import DepthAnything3

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
# 가상 A4 평면 해상도 (DA3 입력용으로 너무 크지 않게 설정)
WARP_W = 518
WARP_H = 732  # A4 비율 (1:1.414)

# 모델 설정
YOLO_PATH = 'finger_project/finger_project/train_result/weights/best.pt'
DA3_MODEL_ID = "depth-anything/DA3-base"  # 또는 로컬 .pth 파일 경로

# 터치 감도 설정 (중요!)
# (손가락 깊이 - 종이 깊이) 차이가 이 값보다 작으면 '터치'로 인정
# DA3 출력은 상대값이므로 테스트하며 조절 필요 (보통 0.05 ~ 0.1 사이)
TOUCH_THRESHOLD = 0.5

# ==========================================
# 2. 초기화
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 디바이스: {device}")

# 모델 로드
yolo_model = YOLO(YOLO_PATH)
try:
    da3_model = DepthAnything3.from_pretrained(DA3_MODEL_ID).to(device).eval()
except Exception as e:
    print(f"DA3 로드 실패: {e}")
    exit()

# ArUco 설정
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_25h9)
parameters = aruco.DetectorParameters()

# 카메라 연결
cap = cv2.VideoCapture(0)
if not cap.isOpened(): exit()


# ==========================================
# 3. 유틸 함수
# ==========================================
def get_perspective_matrix(frame, corners, ids):
    # (이전 코드와 동일한 마커 정렬 및 Matrix 계산 로직)
    ids = ids.flatten()
    corners_map = {id: corner for id, corner in zip(ids, corners)}

    # 0:TL, 1:TR, 3:BR, 2:BL (사용자 마커 배치 기준)
    if not all(i in corners_map for i in [0, 1, 2, 3]):
        return None

    # 바깥쪽 모서리 기준 (최대 영역)
    pt_tl = corners_map[0][0][0]
    pt_tr = corners_map[1][0][1]
    pt_br = corners_map[3][0][2]
    pt_bl = corners_map[2][0][3]

    src_pts = np.array([pt_tl, pt_tr, pt_br, pt_bl], dtype=np.float32)
    dst_pts = np.array([[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]], dtype=np.float32)

    return cv2.getPerspectiveTransform(src_pts, dst_pts)


print("=== 시스템 시작 (종료: q) ===")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. 마커 탐지 및 Warped View 생성
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    matrix = None
    warped_view = None
    depth_map = None

    if ids is not None and len(ids) >= 4:
        matrix = get_perspective_matrix(frame, corners, ids)

        if matrix is not None:
            # [핵심 1] 마커 영역만 잘라서 정면 뷰로 만듦
            warped_view = cv2.warpPerspective(frame, matrix, (WARP_W, WARP_H))

            # [핵심 2] 잘라낸 영역(종이)만 DA3로 깊이 추정
            # (전체 화면보다 작아서 빠르고, 종이 위 물체 분석에 최적화됨)
            try:
                da3_res = da3_model.inference([warped_view])
                depth_map = da3_res.depth[0]  # (WARP_H, WARP_W)

                # 깊이 정규화 (0~1) : 비교를 위해 필수
                d_min, d_max = depth_map.min(), depth_map.max()
                depth_norm = (depth_map - d_min) / (d_max - d_min)
                print(f" - 최소값: {depth_map.min():.4f} (멀다)")
                print(f" - 최대값: {depth_map.max():.4f} (가깝다)")

            except Exception as e:
                print(e)

    # 2. YOLO 손가락 탐지 (원본 프레임에서)
    yolo_results = yolo_model(frame, verbose=False)

    for r in yolo_results:
        for box in r.boxes:
            # 손가락 좌표 (원본)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            fx, fy = (x1 + x2) / 2, y2  # 손가락 끝

            # 시각화 (원본)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # 3. 좌표 변환 및 깊이 비교
            if matrix is not None and depth_map is not None:
                # 손가락 좌표를 Warped View 좌표로 변환
                pts = np.array([[[fx, fy]]], dtype=np.float32)
                transformed = cv2.perspectiveTransform(pts, matrix)
                tx, ty = map(int, transformed[0][0])

                # 범위 체크
                if 0 <= tx < WARP_W and 0 <= ty < WARP_H:
                    # [핵심 3] 손가락 박스 부분의 깊이 vs 종이 깊이 비교
                    # 손가락 끝 주변(ROI)의 깊이값을 가져옴 (예: 10x10 영역)
                    roi_size = 5
                    roi_y1, roi_y2 = max(0, ty - roi_size), min(WARP_H, ty + roi_size)
                    roi_x1, roi_x2 = max(0, tx - roi_size), min(WARP_W, tx + roi_size)

                    finger_depth_roi = depth_norm[roi_y1:roi_y2, roi_x1:roi_x2]

                    if finger_depth_roi.size > 0:
                        finger_z = np.median(finger_depth_roi)  # 손가락 깊이 (중앙값)

                        # 종이(바닥) 깊이는 보통 가장 먼 값(0에 가까움)이거나
                        # 현재 ROI 주변부의 최소값으로 추정 가능
                        # 여기서는 간단히 화면 전체의 하위 10% 값을 바닥으로 가정 (또는 고정값)
                        paper_z = 0.2  # 예시 기준값 (상황에 따라 0.0~0.2 사이)

                        # DA3: 가까울수록 값 큼(1.0), 멀수록 값 작음(0.0)
                        # 손가락이 떠 있음 -> finger_z가 큼 (예: 0.8)
                        # 손가락이 닿음   -> finger_z가 작아짐 (paper_z와 비슷해짐)

                        # 높이 차이 (클수록 떠 있는 것)
                        diff = finger_z - paper_z

                        # 상태 판정
                        if diff < TOUCH_THRESHOLD:
                            status = "TOUCH!"
                            color = (0, 255, 0)  # 초록 (입력)
                        else:
                            status = "Hover"
                            color = (0, 0, 255)  # 빨강 (뜸)

                        # 가상 화면에 표시
                        cv2.circle(warped_view, (tx, ty), 10, color, -1)
                        cv2.putText(warped_view, f"{status} ({diff:.2f})", (tx + 15, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 4. 화면 출력
    cv2.imshow("Original Camera", frame)
    if warped_view is not None:
        # 깊이맵도 같이 보기 (디버깅용)
        depth_vis = (depth_norm * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

        # 가상 화면 + 깊이맵 나란히 출력
        h, w = warped_view.shape[:2]
        depth_vis_resized = cv2.resize(depth_vis, (w, h))
        combined = np.hstack((warped_view, depth_vis_resized))

        cv2.imshow("Warped View & Depth", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()