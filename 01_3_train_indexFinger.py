import os
import shutil
import random
import yaml
from ultralytics import YOLO

# ==============================================================================
# 1. 설정 변수
# ==============================================================================
# 원본 캡쳐 폴더 (검지 데이터만 있다고 가정)
SRC_IMG_DIR = "fingercapture2/image"
SRC_LABEL_DIR = "fingercapture2/label"

# 최종 YOLO 데이터셋 폴더
DST_BASE = "dataset2"
DST_IMG_TRAIN = os.path.join(DST_BASE, "images/train")
DST_IMG_VAL = os.path.join(DST_BASE, "images/val")
DST_LBL_TRAIN = os.path.join(DST_BASE, "labels/train")
DST_LBL_VAL = os.path.join(DST_BASE, "labels/val")

# 필터 파라미터
TRAIN_RATIO = 0.8  # train:val = 8:2
MIN_LABELS_PER_IMAGE = 1  # 최소 박스 개수
MAX_BOX_W = 0.8  # (수정됨) 검지가 크게 찍힐 수 있으므로 0.5 -> 0.8로 완화
MAX_BOX_H = 0.8
MIN_BOX_W = 0.001
MIN_BOX_H = 0.001


# ==============================================================================
# 2. 유틸리티 함수
# ==============================================================================
def make_directories():
    os.makedirs(DST_IMG_TRAIN, exist_ok=True)
    os.makedirs(DST_IMG_VAL, exist_ok=True)
    os.makedirs(DST_LBL_TRAIN, exist_ok=True)
    os.makedirs(DST_LBL_VAL, exist_ok=True)


def create_yaml_file():
    """
    YOLO 학습에 필요한 data.yaml 파일을 자동으로 생성합니다.
    검지(Index Finger) 하나만 학습하므로 nc: 1 입니다.
    """
    yaml_content = {
        'path': os.path.abspath(DST_BASE),  # 절대 경로 사용 추천
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,  # 클래스 개수: 1개 (검지)
        'names': ['index']  # 클래스 이름: index (검지)
    }

    with open('data.yaml', 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print("✅ data.yaml 파일 생성 완료")


def parse_label_file(label_path):
    if not os.path.exists(label_path):
        return None

    labels = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                return None
            try:
                # 검지 데이터만 있더라도, 라벨 파일의 클래스 ID가 0이 아닐 수 있음 (Mediapipe 등 사용 시)
                # 여기서는 읽기만 하고, 학습 시에는 data.yaml 설정에 따라 처리됨
                class_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                return None

            # 0~1 범위 체크
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                return None

            labels.append((class_id, cx, cy, w, h))

    if len(labels) == 0:
        return None

    return labels


def is_good_sample(labels):
    """
    검지만 있는 데이터셋이므로, 단순히 박스가 유효한 범위 내에 있는지만 확인합니다.
    (엄지 체크 로직 제거됨)
    """
    # 최소 박스 개수 체크
    if len(labels) < MIN_LABELS_PER_IMAGE:
        return False

    for class_id, cx, cy, w, h in labels:
        # 너무 큰/작은 박스 제외
        if w < MIN_BOX_W or h < MIN_BOX_H:
            return False
        if w > MAX_BOX_W or h > MAX_BOX_H:
            return False

    return True


def copy_pairs(pairs, img_dst_dir, lbl_dst_dir):
    for img_path, label_path in pairs:
        img_name = os.path.basename(img_path)
        lbl_name = os.path.basename(label_path)

        shutil.copy2(img_path, os.path.join(img_dst_dir, img_name))
        shutil.copy2(label_path, os.path.join(lbl_dst_dir, lbl_name))


# ==============================================================================
# 3. 메인 실행 함수
# ==============================================================================
def main():
    # 1. 디렉토리 초기화 및 생성
    if os.path.exists(DST_BASE):
        try:
            shutil.rmtree(DST_BASE)  # 기존 데이터셋 삭제 후 다시 생성 (꼬임 방지)
        except:
            pass
    make_directories()

    # 2. 데이터 필터링 및 리스트업
    images = sorted(os.listdir(SRC_IMG_DIR))
    selected_pairs = []

    print("데이터 분석 및 필터링 중...")
    for img_name in images:
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(SRC_IMG_DIR, img_name)
        label_path = os.path.join(SRC_LABEL_DIR, base + ".txt")

        labels = parse_label_file(label_path)
        if labels is None:
            continue

        if not is_good_sample(labels):
            continue

        selected_pairs.append((img_path, label_path))

    print(f"총 이미지 수: {len(images)}")
    print(f"학습에 사용할 이미지 수: {len(selected_pairs)}")

    if len(selected_pairs) == 0:
        print("❌ 학습할 데이터가 없습니다. 경로를 확인해주세요.")
        return

    # 3. 데이터 분할 및 복사
    random.shuffle(selected_pairs)
    split_idx = int(len(selected_pairs) * TRAIN_RATIO)
    train_pairs = selected_pairs[:split_idx]
    val_pairs = selected_pairs[split_idx:]

    print("데이터 복사 중...")
    copy_pairs(train_pairs, DST_IMG_TRAIN, DST_LBL_TRAIN)
    copy_pairs(val_pairs, DST_IMG_VAL, DST_LBL_VAL)
    print(f"Train: {len(train_pairs)}장, Val: {len(val_pairs)}장 복사 완료.")

    # 4. data.yaml 파일 생성 (검지 전용)
    create_yaml_file()

    # 5. 모델 훈련 시작
    print("🚀 YOLOv8 모델 훈련 시작...")

    # 처음 실행 시 weights/yolov8n.pt 자동 다운로드
    model = YOLO('yolov8n.pt')

    results = model.train(
        data='data.yaml',  # 위에서 생성한 yaml 파일 사용
        epochs=500,
        imgsz=640,
        batch=16,
        device=0,  # GPU 사용
        workers=0,  # Windows 환경 권장 설정
        project='finger_project2',
        name='index_finger_model',  # 이름을 명확하게 변경
        exist_ok=True,
        patience=20,  # 조기 종료 조건 (20 에포크 동안 향상 없으면 중단)
        verbose=True
    )

    print("🎉 훈련 완료!")


if __name__ == "__main__":
    main()