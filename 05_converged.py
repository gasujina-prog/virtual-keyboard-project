import cv2
import numpy as np
import json
from ultralytics import YOLO

# ==============================
# 0. 설정
# ==============================
CAMERA_INDEX = 0
JSON_PATH = "kblayout.json"
YOLO_WEIGHTS = "finger_project/train_result/weights/best.pt"

DRAW_KEY_BOXES = True
YOLO_CONF = 0.3         # 신뢰도 threshold
YOLO_DEVICE = "cuda:0"

# ==============================
# 1. 키보드 JSON 로드
# ==============================
with open(JSON_PATH, "r", encoding="utf-8") as f:
    kb_layout = json.load(f)

keys = kb_layout["keys"]

# 키보드 정규화 전체 사각형 (0~1)
# 03_keyboard_detecting.py와 동일 순서 유지
kb_quad = np.array([
    [0.0, 0.0],  # BR (label은 틀렸지만 이 순서에 맞춰 튜닝됨)
    [1.0, 0.0],  # BL
    [1.0, 1.0],  # TL
    [0.0, 1.0],  # TR
], dtype=np.float32)


def rotate_keys_180_inplace(keys_list):
    """레이아웃을 180도 회전."""
    for key in keys_list:
        x = key["x"]
        y = key["y"]
        w = key["w"]
        h = key["h"]
        key["x"] = 1.0 - (x + w)
        key["y"] = 1.0 - (y + h)


# 03번에서 사용하던 대로 180도 회전 적용
rotate_keys_180_inplace(keys)


# ==============================
# 2. AprilTag 설정
# ==============================
aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_25h9)
parameters = aruco.DetectorParameters()


def classify_four_tags(corners, ids):
    """4개의 태그를 화면 기준 TL/TR/BL/BR로 분류."""
    ids = np.array(ids).flatten()
    centers = []

    for i, c in enumerate(corners):
        c_flat = np.array(c).reshape(-1, 2)
        cx, cy = c_flat.mean(axis=0)
        centers.append((i, cx, cy))  # index, x, y

    centers = np.array(centers, dtype=float)

    # y 기준 정렬 → 위/아래
    sort_by_y = centers[np.argsort(centers[:, 2])]
    top_two = sort_by_y[:2]
    bottom_two = sort_by_y[2:]

    # 각각 x 기준 정렬 → 좌/우
    top_two = top_two[np.argsort(top_two[:, 1])]
    bottom_two = bottom_two[np.argsort(bottom_two[:, 1])]

    tl_idx = int(top_two[0, 0])
    tr_idx = int(top_two[1, 0])
    bl_idx = int(bottom_two[0, 0])
    br_idx = int(bottom_two[1, 0])

    def corner_inner(idx, role):
        # corners[idx]: [TL, TR, BR, BL]
        c = np.array(corners[idx]).reshape(-1, 2)
        if role == "top_left":
            return c[3]  # BL
        elif role == "top_right":
            return c[2]  # BR
        elif role == "bottom_right":
            return c[1]  # TR
        elif role == "bottom_left":
            return c[0]  # TL
        else:
            return c.mean(axis=0)

    result = {
        "top_left": corner_inner(tl_idx, "top_left"),
        "top_right": corner_inner(tr_idx, "top_right"),
        "bottom_left": corner_inner(bl_idx, "bottom_left"),
        "bottom_right": corner_inner(br_idx, "bottom_right"),
    }
    return result


def draw_keyboard_quad(frame, quad, color=(0, 255, 0)):
    quad_int = quad.astype(int)
    for i in range(4):
        pt1 = tuple(quad_int[i])
        pt2 = tuple(quad_int[(i + 1) % 4])
        cv2.line(frame, pt1, pt2, color, 2)


def select_key_by_center_or_nearest(kx, ky, keys):
    """
    kx, ky: 키보드 정규화 좌표 (0~1)
    1) 이 점을 포함하는 키가 있으면 그 키 반환
    2) 없으면, 중심이 가장 가까운 키를 반환
    """
    inside_key_id = None
    nearest_id = None
    nearest_d2 = 1e9

    for key in keys:
        x_min = key["x"]
        y_min = key["y"]
        x_max = key["x"] + key["w"]
        y_max = key["y"] + key["h"]

        # 안에 들어가면 저장
        if x_min <= kx <= x_max and y_min <= ky <= y_max:
            inside_key_id = key["id"]

        # 가장 가까운 키(center distance)
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        d2 = (cx - kx) ** 2 + (cy - ky) ** 2
        if d2 < nearest_d2:
            nearest_d2 = d2
            nearest_id = key["id"]

    if inside_key_id is not None:
        return inside_key_id
    return nearest_id


# ==============================
# 3. YOLO 손가락 모델 로드
# ==============================
print("[INFO] Loading YOLO fingertip model...")
model = YOLO(YOLO_WEIGHTS)

# ==============================
# 4. 상태 변수
# ==============================
prev_quad = None
prev_tag_centers = {}
H_kb2img = None  # 키보드(0~1) → 이미지
H_img2kb = None  # 이미지 → 키보드(0~1)

# ==============================
# 5. 메인 루프
# ==============================
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --------- AprilTag 탐지 ---------
    corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)
    img_quad = None

    if ids is not None and len(ids) > 0:
        ids = np.array(ids).flatten()
        corners_np = [np.array(c).reshape(-1, 2) for c in corners]

        # 태그 시각화
        aruco.drawDetectedMarkers(frame, corners, ids.reshape(-1, 1))

        curr_tag_centers = {
            int(ids[i]): corners_np[i].mean(axis=0) for i in range(len(ids))
        }
        num_tags = len(ids)

        # 4개 태그 → 완전 homography
        if num_tags >= 4:
            roles = classify_four_tags(corners_np, ids)
            TL = roles["top_left"]
            TR = roles["top_right"]
            BR = roles["bottom_right"]
            BL = roles["bottom_left"]

            img_quad = np.vstack([TL, TR, BR, BL]).astype(np.float32)

            prev_quad = img_quad.copy()
            prev_tag_centers = curr_tag_centers.copy()

        # 3개 태그 → 이전 quad + affine 보정
        elif num_tags == 3 and prev_quad is not None:
            src_pts, dst_pts = [], []
            for tag_id in ids:
                tag_id = int(tag_id)
                if tag_id in prev_tag_centers:
                    src_pts.append(prev_tag_centers[tag_id])
                    dst_pts.append(curr_tag_centers[tag_id])

            if len(src_pts) >= 2:
                src_pts = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
                dst_pts = np.array(dst_pts, dtype=np.float32).reshape(-1, 1, 2)
                M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                if M is not None:
                    prev_quad_h = np.hstack(
                        [prev_quad, np.ones((4, 1), dtype=np.float32)]
                    )
                    img_quad = (M @ prev_quad_h.T).T
                    prev_quad = img_quad.copy()
                    prev_tag_centers = curr_tag_centers.copy()

    # 태그 없음 → 이전 quad 유지
    if img_quad is None and prev_quad is not None:
        img_quad = prev_quad.copy()

    # homography 계산
    if img_quad is not None and img_quad.shape == (4, 2):
        img_quad_f = img_quad.astype(np.float32)

        # 키보드(0~1) → 이미지
        H_kb2img = cv2.getPerspectiveTransform(kb_quad, img_quad_f)
        # 이미지 → 키보드(0~1)
        H_img2kb = cv2.getPerspectiveTransform(img_quad_f, kb_quad)

        # 키보드 외곽선
        draw_keyboard_quad(frame, img_quad_f, color=(0, 255, 0))

        # JSON 키 박스 디버그
        if DRAW_KEY_BOXES and H_kb2img is not None:
            for key in keys:
                x = key["x"]
                y = key["y"]
                w_k = key["w"]
                h_k = key["h"]

                rect_kb = np.array(
                    [
                        [x, y],
                        [x + w_k, y],
                        [x + w_k, y + h_k],
                        [x, y + h_k],
                    ],
                    dtype=np.float32,
                ).reshape(1, -1, 2)

                rect_img = cv2.perspectiveTransform(rect_kb, H_kb2img)[0]
                rect_int = rect_img.astype(int)
                cv2.polylines(frame, [rect_int], True, (80, 0, 0), 1)

    # --------- YOLO 손가락 추론 + 키 매핑 ---------
    if H_img2kb is not None:
        results = model(
            frame, imgsz=640, conf=YOLO_CONF, device=YOLO_DEVICE
        )[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # fingertip 클래스만 사용 (필요 시 thumb 등 추가)
            if cls != 0:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # 1) 중심점을 키보드 좌표계(0~1)로 투영
            pt_img = np.array([[[cx, cy]]], dtype=np.float32)  # (1,1,2)
            pt_kb = cv2.perspectiveTransform(pt_img, H_img2kb)[0][0]
            kx, ky = float(pt_kb[0]), float(pt_kb[1])

            # 2) 키보드 영역 밖이면 no-key
            if not (0.0 <= kx <= 1.0 and 0.0 <= ky <= 1.0):
                key_id = None
            else:
                # 3) 안에 있으면: 그 점 기준으로 키 선택 (없으면 가장 가까운 키)
                key_id = select_key_by_center_or_nearest(kx, ky, keys)

            # -------- 시각화는 항상 수행 --------
            base_color = (0, 255, 255)  # 기본: 노랑
            mapped_color = (0, 255, 0)  # 키 매핑 성공: 초록

            color = mapped_color if key_id is not None else base_color

            # 박스
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2,
            )

            # 중심점
            cv2.circle(
                frame,
                (int(cx), int(cy)),
                6,
                color,
                -1,
            )

            # 라벨 텍스트
            if key_id is not None:
                label = f"{key_id} conf:{conf:.2f}"
            else:
                label = f"no-key conf:{conf:.2f}"

            cv2.putText(
                frame,
                label,
                (int(cx) + 10, int(cy) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    cv2.putText(
        frame,
        "Press 'q' to quit",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    cv2.imshow("Keyboard + YOLO Fingertip", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
