import cv2
import numpy as np
import json
from ultralytics import YOLO

# ==============================
# 0. 설정
# ==============================
CAMERA_INDEX = 0
JSON_PATH = "kblayout.json"
TEXTURE_PATH = "kblayout.png"   # 지금은 안 쓰지만 유지
YOLO_WEIGHTS = "finger_project/train_result/weights/best.pt"  # 02_modeltest.py와 동일 경로

DRAW_KEY_BOXES = True   # 키 영역 디버그용
YOLO_CONF = 0.5         # 신뢰도 threshold
YOLO_DEVICE = "cuda:0"  # GPU 사용, 필요시 "cpu"로 변경

# ==============================
# 1. 키보드 JSON 로드
# ==============================
with open(JSON_PATH, "r", encoding="utf-8") as f:
    kb_layout = json.load(f)

keys = kb_layout["keys"]

# 키보드 정규화 전체 사각형 (0~1)
# (03_keyboard_detecting.py와 동일 순서 유지) :contentReference[oaicite:2]{index=2}
kb_quad = np.array([
    [0.0, 0.0],  # BR (label은 틀렸지만, 이 순서에 맞춰 JSON/튜닝이 되어 있음)
    [1.0, 0.0],  # BL
    [1.0, 1.0],  # TL
    [0.0, 1.0],  # TR
], dtype=np.float32)

def rotate_keys_180_inplace(keys_list):
    for key in keys_list:
        x = key["x"]
        y = key["y"]
        w = key["w"]
        h = key["h"]
        key["x"] = 1.0 - (x + w)
        key["y"] = 1.0 - (y + h)

# 03번에서 이미 사용하던 대로 180도 회전 적용 :contentReference[oaicite:3]{index=3}
rotate_keys_180_inplace(keys)

def find_key_at(kx, ky):
    """정규화 좌표(kx, ky)가 포함되는 키 id 반환 (없으면 None)."""
    for key in keys:
        if (key["x"] <= kx <= key["x"] + key["w"] and
            key["y"] <= ky <= key["y"] + key["h"]):
            return key["id"]
    return None

# ==============================
# 2. (옵션) 키보드 텍스처 이미지 로드
# ==============================
kb_img = cv2.imread(TEXTURE_PATH, cv2.IMREAD_UNCHANGED)
if kb_img is None:
    print(f"[경고] 키보드 텍스처 이미지를 찾을 수 없습니다: {TEXTURE_PATH}")
else:
    kb_img = cv2.rotate(kb_img, cv2.ROTATE_180)  # 03번 코드와 동일 회전

kh, kw = kb_img.shape[:2] if kb_img is not None else (0, 0)

texture_src_pts = np.array([
    [0,   0  ],   # TL
    [kw,  0  ],   # TR
    [kw,  kh ],   # BR
    [0,   kh ]    # BL
], dtype=np.float32) if kb_img is not None else None

# ==============================
# 3. AprilTag 설정
# ==============================
aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_25h9)
parameters = aruco.DetectorParameters()

def classify_four_tags(corners, ids):

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
        c = np.array(corners[idx]).reshape(-1, 2)
        if role == 'top_left':
            return c[3]   # BL
        elif role == 'top_right':
            return c[2]   # BR
        elif role == 'bottom_right':
            return c[1]   # TR
        elif role == 'bottom_left':
            return c[0]   # TL
        else:
            return c.mean(axis=0)

    result = {
        'top_left':      corner_inner(tl_idx, 'top_left'),
        'top_right':     corner_inner(tr_idx, 'top_right'),
        'bottom_left':   corner_inner(bl_idx, 'bottom_left'),
        'bottom_right':  corner_inner(br_idx, 'bottom_right'),
    }
    return result

def draw_keyboard_quad(frame, quad, color=(0, 255, 0)):
    quad_int = quad.astype(int)
    for i in range(4):
        pt1 = tuple(quad_int[i])
        pt2 = tuple(quad_int[(i + 1) % 4])
        cv2.line(frame, pt1, pt2, color, 2)

# ==============================
# 4. YOLO 손가락 모델 로드 (02_modeltest.py 기반) :contentReference[oaicite:5]{index=5}
# ==============================
print("[INFO] Loading YOLO fingertip model...")
model = YOLO(YOLO_WEIGHTS)

def infer_fingers_from_model(frame_bgr):
    """
    YOLO 출력 → 손가락 중심 좌표 리스트로 변환.
    cls 0: fingertip, cls 1: thumb (필요시 둘 다 사용 가능).
    """
    results = model(frame_bgr, imgsz=640, conf=YOLO_CONF, device=YOLO_DEVICE)[0]

    fingers = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # 여기서는 일단 fingertip(0)만 사용 (thumb은 추후 확장)
        if cls != 0:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        fingers.append({
            "pt": (cx, cy),
            "conf": conf,
            "cls": cls
        })

    return fingers

# ==============================
# 5. 상태 변수
# ==============================
prev_quad = None
prev_tag_centers = {}
H_norm = None
H_norm_inv = None

# ==============================
# 6. 메인 루프
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
            int(ids[i]): corners_np[i].mean(axis=0)
            for i in range(len(ids))
        }
        num_tags = len(ids)

        # 4개 태그 → 완전 homography
        if num_tags >= 4:
            roles = classify_four_tags(corners_np, ids)
            TL = roles['top_left']
            TR = roles['top_right']
            BR = roles['bottom_right']
            BL = roles['bottom_left']

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
        draw_keyboard_quad(frame, img_quad, color=(0, 255, 0))

        H_norm, _ = cv2.findHomography(kb_quad, img_quad)
        if H_norm is not None:
            H_norm_inv = np.linalg.inv(H_norm)

        if DRAW_KEY_BOXES and H_norm is not None:
            for key in keys:
                x = key["x"]; y = key["y"]
                w_k = key["w"]; h_k = key["h"]

                rect_kb = np.array([
                    [x,        y       ],
                    [x + w_k,  y       ],
                    [x + w_k,  y + h_k ],
                    [x,        y + h_k ]
                ], dtype=np.float32)

                rect_img = cv2.perspectiveTransform(rect_kb[None, :, :], H_norm)[0]
                rect_int = rect_img.astype(int)
                cv2.polylines(frame, [rect_int], True, (80, 0, 0), 1)

    # --------- YOLO 손가락 추론 + 키 매핑 ---------
    if H_norm_inv is not None:
        fingers = infer_fingers_from_model(frame)

        for f in fingers:
            cx, cy = f["pt"]
            conf = f["conf"]

            # 이미지 → 키보드 정규화 좌표
            pt_img = np.array([[ [cx, cy] ]], dtype=np.float32)
            pt_kb = cv2.perspectiveTransform(pt_img, H_norm_inv)[0][0]
            kx, ky = float(pt_kb[0]), float(pt_kb[1])

            # 키보드 영역 밖이면 무시
            if not (0.0 <= kx <= 1.0 and 0.0 <= ky <= 1.0):
                continue

            key_id = find_key_at(kx, ky)

            # 시각화
            color = (0, 255, 255)
            cv2.circle(frame, (int(cx), int(cy)), 6, color, -1)

            if key_id is not None:
                cv2.putText(frame, f"{key_id} ({conf:.2f})",
                            (int(cx) + 10, int(cy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    else:
        cv2.putText(frame, "NO KEYBOARD HOMOGRAPHY",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2)

    cv2.putText(frame, "Press 'q' to quit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Keyboard + YOLO Fingertip", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
