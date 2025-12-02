import cv2
import numpy as np

# -------------------------------
# 1. 유틸 함수
# -------------------------------

def classify_four_tags(corners, ids):
    """
    corners: list of (4,2) ndarray
    ids    : (N,1) or (N,)
    화면 기준으로 TL/TR/BL/BR 분류
    """
    ids = np.array(ids).flatten()
    centers = []

    for i, c in enumerate(corners):
        c_flat = np.array(c).reshape(-1, 2)
        cx, cy = c_flat.mean(axis=0)
        centers.append((i, cx, cy))  # (index, x, y)

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
            return c[3]
        elif role == 'top_right':
            return c[2]
        elif role == 'bottom_right':
            return c[1]
        elif role == 'bottom_left':
            return c[0]
        else:
            # 혹시 모를 경우 센터 fallback
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


# -------------------------------
# 2. AprilTag (25h9) 설정
# -------------------------------

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_25h9)
parameters = aruco.DetectorParameters()

# -------------------------------
# 3. 상태 변수
# -------------------------------

prev_quad = None              # 이전 프레임의 키보드 4점 (4,2)
prev_tag_centers = {}         # 이전 프레임의 각 태그 중심 {id: (x,y)}

# -------------------------------
# 4. 웹캠 루프
# -------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    raise SystemExit

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)

    if ids is not None and len(ids) > 0:
        # numpy 형태 정리
        ids = np.array(ids).flatten()
        corners_np = [np.array(c).reshape(-1, 2) for c in corners]

        # 태그 시각화
        aruco.drawDetectedMarkers(frame, corners, ids.reshape(-1, 1))

        # 현재 프레임의 태그 중심 좌표 저장
        curr_tag_centers = {}
        for i, tag_id in enumerate(ids):
            c = corners_np[i]
            center = c.mean(axis=0)
            curr_tag_centers[int(tag_id)] = center

        num_tags = len(ids)

        img_quad = None

        # ---------- 4개 태그 다 보이는 경우: 정답 homography ----------
        if num_tags >= 4:
            roles = classify_four_tags(corners_np, ids)
            TL = roles['top_left']
            TR = roles['top_right']
            BR = roles['bottom_right']
            BL = roles['bottom_left']

            img_quad = np.vstack([TL, TR, BR, BL]).astype(np.float32)

            # 키보드 로컬 좌표 (0~1 정규화 사각형)
            kb_quad = np.array([
                [0.0, 0.0],  # TL
                [1.0, 0.0],  # TR
                [1.0, 1.0],  # BR
                [0.0, 1.0],  # BL
            ], dtype=np.float32)

            H, _ = cv2.findHomography(kb_quad, img_quad)

            # 상태 업데이트
            prev_quad = img_quad.copy()
            prev_tag_centers = curr_tag_centers.copy()

        # ---------- 3개 태그만 보이는 경우: 시간축 활용해서 보정 ----------
        elif num_tags == 3 and prev_quad is not None and len(prev_tag_centers) > 0:
            # 이전 프레임의 같은 태그와 매칭
            src_pts = []
            dst_pts = []
            for tag_id in ids:
                tag_id = int(tag_id)
                if tag_id in prev_tag_centers:
                    src_pts.append(prev_tag_centers[tag_id])   # 이전 위치
                    dst_pts.append(curr_tag_centers[tag_id])   # 현재 위치

            src_pts = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
            dst_pts = np.array(dst_pts, dtype=np.float32).reshape(-1, 1, 2)

            if len(src_pts) >= 2:
                # 회전 + 스케일 + 이동 (affine) 추정
                M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                if M is not None:
                    # 이전 quad 전체에 동일 변환 적용
                    prev_quad_h = np.hstack([
                        prev_quad, np.ones((4, 1), dtype=np.float32)
                    ])  # (4,3)
                    img_quad = (M @ prev_quad_h.T).T  # (4,2)

                    prev_quad = img_quad.copy()
                    prev_tag_centers = curr_tag_centers.copy()

        # ---------- 2개 이하거나, 추정 실패 → 이전 quad 그냥 유지 ----------
        if img_quad is None and prev_quad is not None:
            img_quad = prev_quad.copy()

        # ---------- quad 그리기 ----------
        if img_quad is not None and img_quad.shape == (4, 2):
            draw_keyboard_quad(frame, img_quad)

    cv2.putText(frame, "Press 'q' to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("AprilTag Keyboard Test (3-tag + time)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
