import cv2
import cv2.aruco as aruco
import numpy as np
import json

# ==========================================
# 1. 설정 (기존 프로젝트와 동일하게 맞춤)
# ==========================================
WARP_W = 1200  # 평면으로 펼쳤을 때 가로 크기
WARP_H = 620  # 평면으로 펼쳤을 때 세로 크기
OUTPUT_FILE = "../../key_layout2.json"

# ArUco 설정 (기존 코드와 동일한 딕셔너리 사용)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_25h9)
parameters = aruco.DetectorParameters()

# ==========================================
# 2. 전역 변수
# ==========================================
drawing = False
ix, iy = -1, -1
rects = {}  # { "KeyName": [x, y, w, h] }
captured_img = None
display_img = None


# ==========================================
# 3. 마우스 콜백 함수 (드래그로 영역 지정)
# ==========================================
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, captured_img, display_img, rects

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            display_img = captured_img.copy()
            # 기존에 그린 사각형들도 계속 보여주기
            for key, (rx, ry, rw, rh) in rects.items():
                cv2.rectangle(display_img, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
                cv2.putText(display_img, key, (rx, ry - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # 지금 그리고 있는 사각형 (초록색 점선 느낌)
            cv2.rectangle(display_img, (ix, iy), (x, y), (0, 255, 255), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # 좌표 정규화 (거꾸로 드래그했을 때 대비)
        x_start = min(ix, x)
        y_start = min(iy, y)
        w = abs(ix - x)
        h = abs(iy - y)

        if w < 5 or h < 5: return  # 너무 작으면 무시

        # 화면 업데이트
        display_img = captured_img.copy()
        for key, (rx, ry, rw, rh) in rects.items():
            cv2.rectangle(display_img, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            cv2.putText(display_img, key, (rx, ry - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 현재 그린 박스 표시 (파란색)
        cv2.rectangle(display_img, (x_start, y_start), (x_start + w, y_start + h), (255, 0, 0), 2)
        cv2.imshow("Layout Editor", display_img)
        cv2.waitKey(1)

        # ★ 키 이름 입력 받기
        print(f"\n📍 영역 지정됨: ({x_start}, {y_start}, {w}, {h})")
        key_name = input("⌨️ 키 이름 입력 (취소: Enter): ").strip()

        if key_name:
            rects[key_name] = [x_start, y_start, w, h]
            print(f"✅ 추가됨: {key_name}")
            # 확정된 박스 덮어쓰기 (영구 표시)
            cv2.rectangle(captured_img, (x_start, y_start), (x_start + w, y_start + h), (0, 255, 0), 2)
            cv2.putText(captured_img, key_name, (x_start+10, y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            display_img = captured_img.copy()
        else:
            print("❌ 취소됨")
            display_img = captured_img.copy()


# ==========================================
# 4. 메인 코드
# ==========================================
def main():
    global captured_img, display_img

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        exit()

    print("=== 1단계: 카메라 위치 잡기 ===")
    print("👉 마커 4개가 모두 인식되면 화면이 펴집니다.")
    print("👉 [Space]: 현재 화면 캡처 및 편집 모드 시작")
    print("👉 [q]: 종료")

    # [Step 1] 카메라 루프 (Warping 확인)
    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        warped_view = np.zeros((WARP_H, WARP_W, 3), dtype=np.uint8)
        matrix = None

        if ids is not None and len(ids) >= 4:
            ids = ids.flatten()
            # ID별 코너 매핑
            corners_map = {id: corner for id, corner in zip(ids, corners)}

            # 0,1,2,3번 마커가 다 있어야 함 (사용자 환경에 맞게 ID 수정 가능)
            if all(i in corners_map for i in [0, 1, 2, 3]):
                try:
                    # 좌표 순서: TL(0), TR(1), BR(3), BL(2) - 사용자 마커 배치에 따름
                    # (일반적으로 좌상=0, 우상=1, 우하=2, 좌하=3이 아니라 배치에 따라 다를 수 있음)
                    # 여기서는 0:TL, 1:TR, 3:BR, 2:BL 순서로 가정 (기존 코드 참고)
                    src_pts = np.array([
                        corners_map[0][0][1],  # TL
                        corners_map[1][0][0],  # TR
                        corners_map[3][0][3],  # BR
                        corners_map[2][0][2]  # BL
                    ], dtype=np.float32)

                    dst_pts = np.array([
                        [0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]
                    ], dtype=np.float32)

                    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    warped_view = cv2.warpPerspective(frame, matrix, (WARP_W, WARP_H))

                    # 가이드라인 표시
                    cv2.putText(warped_view, "Press SPACE to Capture", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                except Exception as e:
                    pass

        # 화면 출력
        cv2.imshow("Camera View (Raw)", frame)
        cv2.imshow("Warped View (Result)", warped_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if matrix is not None:
                captured_img = warped_view.copy()  # 캡처!
                display_img = captured_img.copy()
                break
            else:
                print("⚠️ 마커 4개가 인식되지 않아 캡처할 수 없습니다.")
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    # [Step 2] 편집 루프 (마우스 그리기)
    print("\n=== 2단계: 레이아웃 편집 모드 ===")
    print("👉 마우스로 키 영역을 드래그하세요.")
    print("👉 콘솔창에 키 이름을 입력하세요.")
    print("👉 [s]: 저장 후 종료")
    print("👉 [q]: 저장하지 않고 종료")

    cv2.namedWindow("Layout Editor")
    cv2.setMouseCallback("Layout Editor", draw_rectangle)

    while True:
        cv2.imshow("Layout Editor", display_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if not rects:
                print("⚠️ 저장할 데이터가 없습니다.")
                continue

            with open(OUTPUT_FILE, "w", encoding='utf-8') as f:
                json.dump(rects, f, indent=4, ensure_ascii=False)
            print(f"\n💾 '{OUTPUT_FILE}' 파일로 저장되었습니다!")
            break

        elif key == ord('q'):
            print("\n👋 종료합니다.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()