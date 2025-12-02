import cv2
import numpy as np
import json

# ==============================
# 0. 설정
# ==============================
CAMERA_INDEX = 0
JSON_PATH = "kblayout.json"
TEXTURE_PATH = "kblayout.png"

DRAW_KEY_BOXES = True  # True로 바꾸면 각 키 영역도 같이 그려줌(디버그용)

# ==============================
# 1. 키보드 JSON 로드
# ==============================
with open(JSON_PATH, "r", encoding="utf-8") as f:
    kb_layout = json.load(f)

keys = kb_layout["keys"]

# 키보드 정규화 전체 사각형 (0~1)
kb_quad = np.array([
    [0.0, 0.0],  # TL
    [1.0, 0.0],  # TR
    [1.0, 1.0],  # BR
    [0.0, 1.0],  # BL
], dtype=np.float32)


# (필요하면 여기에서 한 번만 180도 회전)
def rotate_keys_180_inplace(keys_list):
    for key in keys_list:
        x = key["x"]
        y = key["y"]
        w = key["w"]
        h = key["h"]
        key["x"] = 1.0 - (x + w)
        key["y"] = 1.0 - (y + h)

# 예) 180도 뒤집고 싶으면 아래 주석 해제
rotate_keys_180_inplace(keys)

# ==============================
# 2. 키보드 텍스처 이미지 로드
# ==============================
kb_img = cv2.imread(TEXTURE_PATH, cv2.IMREAD_UNCHANGED)
if kb_img is None:
    raise FileNotFoundError(f"키보드 텍스처 이미지를 찾을 수 없습니다: {TEXTURE_PATH}")
kb_img = cv2.rotate(kb_img, cv2.ROTATE_180)
kh, kw = kb_img.shape[:2]

# 텍스처 이미지의 4코너 (소스 포인트)
texture_src_pts = np.array([
    [0,   0  ],   # TL
    [kw,  0  ],   # TR
    [kw,  kh ],   # BR
    [0,   kh ]    # BL
], dtype=np.float32)

# ==============================
# 3. AprilTag 설정
# ==============================
aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_25h9)
parameters = aruco.DetectorParameters()

def classify_four_tags(corners, ids):
    """
    corners: list of (4,2)
    ids    : (N,)
    화면 기준으로 TL/TR/BL/BR 분류 후,
    각 태그의 '안쪽' 꼭짓점 사용 (존님이 지정한 방식)
    """
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
        """
        corners[idx]: [TL, TR, BR, BL]
        키보드 안쪽을 향하는 꼭짓점 선택
        """
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
# 4. 상태 변수
# ==============================
prev_quad = None
prev_tag_centers = {}

# ==============================
# 5. 웹캠 루프
# ==============================
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)

    img_quad = None

    if ids is not None and len(ids) > 0:
        ids = np.array(ids).flatten()
        corners_np = [np.array(c).reshape(-1, 2) for c in corners]

        # 태그 시각화
        aruco.drawDetectedMarkers(frame, corners, ids.reshape(-1, 1))

        # 현재 프레임 태그 중심
        curr_tag_centers = {
            int(ids[i]): corners_np[i].mean(axis=0)
            for i in range(len(ids))
        }
        num_tags = len(ids)

        # ---------- 4개 태그: 완전 homography ----------
        if num_tags >= 4:
            roles = classify_four_tags(corners_np, ids)
            TL = roles['top_left']
            TR = roles['top_right']
            BR = roles['bottom_right']
            BL = roles['bottom_left']

            img_quad = np.vstack([TL, TR, BR, BL]).astype(np.float32)

            prev_quad = img_quad.copy()
            prev_tag_centers = curr_tag_centers.copy()

        # ---------- 3개 태그: 이전 quad + affine 보정 ----------
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

        # ---------- 태그 부족: 이전 quad 유지 ----------
        if img_quad is None and prev_quad is not None:
            img_quad = prev_quad.copy()

        # ---------- 키보드 텍스처 + (옵션) 키 박스 시각화 ----------
        if img_quad is not None and img_quad.shape == (4, 2):
            # 큰 초록 박스(디버그용)
            draw_keyboard_quad(frame, img_quad, color=(0, 255, 0))



            if DRAW_KEY_BOXES:
                # 정규화 키보드 → 이미지 homography
                H_norm, _ = cv2.findHomography(kb_quad, img_quad)
                if H_norm is not None:
                    for key in keys:
                        x = key["x"]
                        y = key["y"]
                        w_k = key["w"]
                        h_k = key["h"]

                        rect_kb = np.array([
                            [x, y],
                            [x + w_k, y],
                            [x + w_k, y + h_k],
                            [x, y + h_k]
                        ], dtype=np.float32)

                        rect_img = cv2.perspectiveTransform(rect_kb[None, :, :], H_norm)[0]
                        rect_int = rect_img.astype(int)
                        cv2.polylines(frame, [rect_int], True, (0, 0, 50), 1)



    cv2.putText(frame, "Press 'q' to quit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Keyboard Texture Warp (Full Version)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
