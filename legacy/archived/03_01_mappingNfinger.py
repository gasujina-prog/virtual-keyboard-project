import cv2
import cv2.aruco as aruco
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import time
import pyautogui
from ultralytics import YOLO

# ==========================================
# 1. 설정 및 상수
# ==========================================
# 맵핑할 때 사용했던 해상도와 동일해야 합니다. (JSON 파일 만들 때의 해상도)
# 보내주신 JSON 좌표를 보니 가로가 1200 정도까지 가는 것 같아 1200x620으로 맞췄습니다.
WARP_W = 1200
WARP_H = 620
LAYOUT_FILE = "key_layout.json"
MODEL_PATH = R'finger_project\finger_project\train_result\weights\YUN_best.pt'  # 모델 경로 확인 필수!

# 입력 설정
DWELL_TIME_THRESHOLD = 0.5  # 0.5초 머무르면 입력
COOLDOWN_TIME = 0.3  # 입력 후 0.3초 대기
MULTI_FINGER_MODE = True  # 여러 손가락 동시 입력 허용

# pyautogui 안전장치 (마우스가 구석으로 가면 강제 종료)
pyautogui.FAILSAFE = True

# 특수 키 매핑 (JSON의 키 이름 -> pyautogui 키 이름)
SPECIAL_KEYS = {
    "SpaceBar": "space",
    "Enter": "enter",
    "Backspace": "backspace",
    "Tab": "tab",
    "CapsLock": "capslock",
    "Shift": "shift",
    "RShift": "shiftright",
    "Ctrl": "ctrl",
    "RCtrl": "ctrlright",
    "Alt": "alt",
    "RAlt": "altright",
    "Win": "win",
    "한/영": "한/영",  # Fn키는 OS에서 직접 제어하기 어려움
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "Home": "home",
    "End": "end",
    "PageUp": "pageup",
    "PageDown": "pagedown",
    "~": "`",  # 물결표는 백틱으로 매핑
    "\\": "\\"
}

# JSON 로드
try:
    with open(LAYOUT_FILE, "r", encoding="utf-8") as f:
        raw_layout = json.load(f)

    # JSON 포맷 변환 (리스트 -> 딕셔너리)
    # 파일에는 "key": [x, y, w, h] 로 저장되어 있음
    KEY_LAYOUT = {}
    for k, v in raw_layout.items():
        KEY_LAYOUT[k] = {'x': v[0], 'y': v[1], 'w': v[2], 'h': v[3]}

    print(f"✅ {LAYOUT_FILE} 로드 성공! 키 개수: {len(KEY_LAYOUT)}")
except FileNotFoundError:
    print(f"❌ {LAYOUT_FILE} 파일이 없습니다. 같은 폴더에 넣어주세요.")
    exit()

# YOLO 로드
print("모델 로딩 중...")
model = YOLO(MODEL_PATH)

# AprilTag 설정
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_25h9)
parameters = aruco.DetectorParameters()


# ==========================================
# 2. 유틸 함수 (마커 분류 및 추적)
# ==========================================
def classify_four_tags(corners, ids):
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

    # ★ 바깥쪽 모서리 선택 로직 (맵핑 범위 최대화) ★
    # AprilTag 코너 인덱스: 0=좌상, 1=우상, 2=우하, 3=좌하
    def corner_outer(idx, role):
        c = np.array(corners[idx]).reshape(-1, 2)
        if role == 'TL':
            return c[3]  # 좌상단의 좌상
        elif role == 'TR':
            return c[2]  # 우상단의 우상
        elif role == 'BR':
            return c[1]  # 우하단의 우하
        elif role == 'BL':
            return c[0]  # 좌하단의 좌하
        return c.mean(axis=0)

    return {
        'TL': corner_outer(tl_idx, 'TL'),
        'TR': corner_outer(tr_idx, 'TR'),
        'BL': corner_outer(bl_idx, 'BL'),
        'BR': corner_outer(br_idx, 'BR')
    }


def draw_keyboard_quad(frame, quad, color=(0, 255, 0)):
    quad_int = quad.astype(int)
    for i in range(4):
        cv2.line(frame, tuple(quad_int[i]), tuple(quad_int[(i + 1) % 4]), color, 2)


def draw_keyboard_text_all(img, layout_dict, font_path="malgun.ttf", size=20):
    """
    이미지를 Pillow로 변환한 뒤, 모든 키의 글자를 한 번에 쓰고 다시 OpenCV 포맷으로 반환
    """
    # 1. OpenCV(BGR) -> PIL(RGB) 변환 (딱 한 번!)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype(font_path, size)
    except:
        font = ImageFont.load_default()

    # 2. 모든 키에 대해 반복해서 글씨 쓰기 (변환 없이 그리기만 반복)
    for key, rect in layout_dict.items():
        # rect 포맷 확인 (리스트 or 딕셔너리)
        if isinstance(rect, dict):
            x, y = rect['x'], rect['y']
        else:
            x, y = rect[0], rect[1]

        # 글자 위치 잡기 (박스 안쪽)
        draw.text((x + 5, y + 5), key, font=font, fill=(0, 255, 0))  # 검은색 글씨

    # 3. PIL(RGB) -> OpenCV(BGR) 변환 (딱 한 번!)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ==========================================
# 3. 상태 변수
# ==========================================
prev_homography = None
prev_quad = None
prev_tag_centers = {}

# 키별 상태 관리: { 'key_name': { 'start_time': 0.0, 'last_input': 0.0 } }
key_states = {k: {'start_time': 0, 'last_input': 0} for k in KEY_LAYOUT.keys()}

# ==========================================
# 4. 메인 루프
# ==========================================
cap = cv2.VideoCapture(0)
if not cap.isOpened(): exit()

print("=== 가상 키보드 시작 (종료: q) ===")

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    current_homography = None
    warped_view = np.zeros((WARP_H, WARP_W, 3), dtype=np.uint8)

    # [1] 마커 추적 및 변환 행렬 계산
    corners_np = []
    if ids is not None:
        ids = ids.flatten()
        corners_np = [np.array(c).reshape(-1, 2) for c in corners]
        # aruco.drawDetectedMarkers(frame, corners, ids.reshape(-1, 1)) # (선택) 마커 테두리 보기

    curr_tag_centers = {}
    if ids is not None:
        for i, tag_id in enumerate(ids):
            curr_tag_centers[int(tag_id)] = corners_np[i].mean(axis=0)

    num_tags = len(ids) if ids is not None else 0
    img_quad = None

    # A. 4개 다 보임 (기준 갱신)
    if num_tags >= 4:
        roles = classify_four_tags(corners_np, ids)
        src_pts = np.array([roles['BR'], roles['BL'], roles['TL'], roles['TR']], dtype=np.float32)
        dst_pts = np.array([[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]], dtype=np.float32)

        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        current_homography = H

        prev_homography = H.copy()
        prev_tag_centers = curr_tag_centers.copy()
        img_quad = src_pts.copy()
        prev_quad = img_quad.copy()

    # B. 3개 이하 (추적)
    elif num_tags >= 2 and prev_homography is not None:
        common_ids = []
        src_prev, src_curr = [], []
        for tid in curr_tag_centers:
            if tid in prev_tag_centers:
                common_ids.append(tid)
                src_prev.append(prev_tag_centers[tid])
                src_curr.append(curr_tag_centers[tid])

        if len(common_ids) >= 2:
            src_prev = np.array(src_prev).reshape(-1, 1, 2)
            src_curr = np.array(src_curr).reshape(-1, 1, 2)
            M, _ = cv2.estimateAffinePartial2D(src_prev, src_curr)

            if M is not None:
                M_homo = np.eye(3)
                M_homo[:2] = M
                prev_quad_h = np.hstack([prev_quad, np.ones((4, 1))])
                curr_quad = (prev_quad_h @ M_homo.T)[:, :2].astype(np.float32)

                dst_pts = np.array([[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]], dtype=np.float32)
                current_homography = cv2.getPerspectiveTransform(curr_quad, dst_pts)
                img_quad = curr_quad

    # C. 추적 실패 (유지)
    if current_homography is None and prev_homography is not None:
        current_homography = prev_homography
        img_quad = prev_quad

    if img_quad is not None:
        draw_keyboard_quad(frame, img_quad)

    # [2] 가상 화면 생성 (키보드 그리기)
    if current_homography is not None:
        # 워핑된 화면은 검은 배경으로 시작 (리소스 절약)
        # 키보드 레이아웃 그리기
        # 1. 먼저 박스(사각형)만 OpenCV로 빠르게 다 그립니다.
        for key, rect in KEY_LAYOUT.items():
            if isinstance(rect, dict):
                rx, ry, rw, rh = rect['x'], rect['y'], rect['w'], rect['h']
            else:
                rx, ry, rw, rh = rect
            # 박스 그리기 (OpenCV가 더 빠름)
            cv2.rectangle(warped_view, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
        # 2. 글씨는 한방에 몰아서 그립니다. (변환 비용 최소화)
        warped_view = draw_keyboard_text_all(warped_view, KEY_LAYOUT)

    # [3] YOLO 손가락 탐지
    results = model(frame, verbose=False)
    fingers = []
    for r in results:
        for box in r.boxes:
            # box.cls를 확인하여 손가락(1번)인지 체크하는 것이 좋음 (현재는 모든 객체)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 손가락 끝 좌표 추정 (박스 하단 중앙)
            # 엄지/검지 구분이 있다면 여기서 처리 가능
            fx = (x1 + x2) / 2
            fy = (y1 - y2) / 3 + y2
            fingers.append((fx, fy))

            # 원본 화면 표시
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.circle(frame, (int(fx), int(fy)), 5, (0, 0, 255), -1)

    # [4] 좌표 변환 및 키 입력
    if current_homography is not None and fingers:
        # 좌표 변환
        fingers_np = np.array([fingers], dtype=np.float32).transpose(1, 0, 2)
        transformed_fingers = cv2.perspectiveTransform(fingers_np, current_homography)

        active_keys = set()
        curr_time = time.time()

        for pt in transformed_fingers:
            tx, ty = pt[0]

            # 가상 화면에 손가락 표시
            cv2.circle(warped_view, (int(tx), int(ty)), 8, (0, 0, 255), -1)

            # 히트 테스트
            # ---------------------------------------------------------
            # [수정] 1. OpenCV로 박스(Rectangle) 먼저 그리기 (빠름)
            # ---------------------------------------------------------
            for key_name, rect in KEY_LAYOUT.items():
                rx, ry, rw, rh = rect['x'], rect['y'], rect['w'], rect['h']

                # 기본 스타일 (안 눌림)
                color = (0, 255, 0)  # 초록색
                thickness = 1

                # 히트 테스트 (손가락이 키 안에 있는지 확인)
                if rx < tx < rx + rw and ry < ty < ry + rh:
                    active_keys.add(key_name)
                    color = (0, 255, 255)  # 노란색 (눌림)
                    thickness = -1  # 채우기

                # 박스 그리기
                cv2.rectangle(warped_view, (rx, ry), (rx + rw, ry + rh), color, thickness)

            # ---------------------------------------------------------
            # [수정] 2. Pillow로 글씨(Text) 한 번에 쓰기 (한글/특수문자 지원)
            # ---------------------------------------------------------
            # (1) OpenCV(BGR) -> Pillow(RGB) 변환
            img_pil = Image.fromarray(cv2.cvtColor(warped_view, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            # 폰트 설정 (없으면 기본 폰트)
            try:
                font = ImageFont.truetype("malgun.ttf", 20)  # 윈도우 맑은 고딕
            except:
                font = ImageFont.load_default()

            # (2) 모든 키의 글씨 쓰기
            for key_name, rect in KEY_LAYOUT.items():
                rx, ry = rect['x'], rect['y']
                # 글씨 색상 (검정)
                draw.text((rx + 5, ry + 20), key_name, font=font, fill=(0, 0, 0))

            # (3) Pillow(RGB) -> OpenCV(BGR) 복구
            warped_view = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        # 입력 로직 (상태 머신)
        for key in active_keys:
            state = key_states[key]

            # 처음 눌림
            if state['start_time'] == 0:
                state['start_time'] = curr_time

            # 체류 시간
            duration = curr_time - state['start_time']

            # 입력 확정
            if duration > DWELL_TIME_THRESHOLD:
                if curr_time - state['last_input'] > COOLDOWN_TIME:
                    print(f"👉 Press: {key}")

                    # 특수 키 처리
                    py_key = SPECIAL_KEYS.get(key, key.lower())

                    if py_key:  # None이 아니면 입력
                        try:
                            pyautogui.press(py_key)
                        except:
                            print(f"입력 불가 키: {key}")

                    state['last_input'] = curr_time
                    # 연타 방지를 위해 start_time은 유지하지 않고 리셋하려면 아래 주석 해제
                    state['start_time'] = 0

                    # 입력 피드백 (빨간색)
                    rx, ry, rw, rh = KEY_LAYOUT[key]['x'], KEY_LAYOUT[key]['y'], KEY_LAYOUT[key]['w'], KEY_LAYOUT[key][
                        'h']
                    cv2.rectangle(warped_view, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), -1)

        # 안 눌린 키 리셋
        for key in KEY_LAYOUT:
            if key not in active_keys:
                key_states[key]['start_time'] = 0

    # [5] 화면 출력
    cv2.imshow("Tracking Cam", frame)
    cv2.imshow("Virtual Keyboard", warped_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()