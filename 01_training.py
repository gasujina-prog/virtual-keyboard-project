import os
import shutil
import random
# 먼저 기존 torch가 있다면 삭제 (충돌 방지)
#pip uninstall torch torchvision torchaudio -y

# CUDA 12.1 버전용 PyTorch 설치 (추천)
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
from ultralytics import YOLO
import torch

# 원본 캡쳐 폴더
SRC_IMG_DIR = "fingercapture/image"
SRC_LABEL_DIR = "fingercapture/label"

# 최종 YOLO 데이터셋 폴더
DST_BASE = "dataset"
DST_IMG_TRAIN = os.path.join(DST_BASE, "images/train")
DST_IMG_VAL   = os.path.join(DST_BASE, "images/val")
DST_LBL_TRAIN = os.path.join(DST_BASE, "labels/train")
DST_LBL_VAL   = os.path.join(DST_BASE, "labels/val")

os.makedirs(DST_IMG_TRAIN, exist_ok=True)
os.makedirs(DST_IMG_VAL, exist_ok=True)
os.makedirs(DST_LBL_TRAIN, exist_ok=True)
os.makedirs(DST_LBL_VAL, exist_ok=True)

# 필터 파라미터
TRAIN_RATIO = 0.8          # train:val = 8:2
MIN_LABELS_PER_IMAGE = 1   # 최소 박스 개수
REQUIRE_THUMB = False       # 엄지(클래스 1)가 반드시 포함된 이미지만 사용할지 여부
MAX_BOX_W = 0.5            # 너무 큰 박스 (프레임의 절반 이상)는 제외
MAX_BOX_H = 0.5
MIN_BOX_W = 0.01           # 너무 작은 박스(1% 미만) 제거
MIN_BOX_H = 0.01

def parse_label_file(label_path):
    """
    YOLO 형식: class cx cy w h
    정상적이면 (class_id, cx, cy, w, h) 리스트 반환
    비정상이면 None 반환
    """
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
                # 형식 이상
                return None
            try:
                class_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w  = float(parts[3])
                h  = float(parts[4])
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
    라벨 리스트가 '좋은' 샘플인지 판단하는 함수.
    여기서 품질 기준을 설정.
    """
    # 최소 박스 개수
    if len(labels) < MIN_LABELS_PER_IMAGE:
        return False

    # 박스 크기 체크
    has_thumb = False
    for class_id, cx, cy, w, h in labels:
        # 너무 큰/작은 박스 제외
        if w < MIN_BOX_W or h < MIN_BOX_H:
            return False
        if w > MAX_BOX_W or h > MAX_BOX_H:
            return False

        if class_id == 1:  # 엄지
            has_thumb = True

    if REQUIRE_THUMB and not has_thumb:
        return False

    return True


def main():
    images = sorted(os.listdir(SRC_IMG_DIR))
    selected_pairs = []

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
    print(f"선택된(필터 통과) 이미지 수: {len(selected_pairs)}")

    # 셔플 후 train/val 분할
    random.shuffle(selected_pairs)
    split_idx = int(len(selected_pairs) * TRAIN_RATIO)
    train_pairs = selected_pairs[:split_idx]
    val_pairs = selected_pairs[split_idx:]

    # 복사
    def copy_pairs(pairs, img_dst_dir, lbl_dst_dir):
        for img_path, label_path in pairs:
            img_name = os.path.basename(img_path)
            lbl_name = os.path.basename(label_path)

            shutil.copy2(img_path, os.path.join(img_dst_dir, img_name))
            shutil.copy2(label_path, os.path.join(lbl_dst_dir, lbl_name))

    copy_pairs(train_pairs, DST_IMG_TRAIN, DST_LBL_TRAIN)
    copy_pairs(val_pairs, DST_IMG_VAL,   DST_LBL_VAL)

    print(f"train: {len(train_pairs)}개, val: {len(val_pairs)}개 복사 완료.")

    # CUDA GPU 사용해서 훈련하기
    # 2. 모델 불러오기
    # yolov8n.pt (nano): 가장 가볍고 빠름 / yolov8s.pt (small), yolov8m.pt (medium) 등 선택 가능
    # 처음 실행 시 가중치 파일을 자동으로 다운로드합니다.
    model = YOLO('yolov8n.pt')

    # 3. 모델 훈련
    results = model.train(
        data='data.yaml',  # 앞서 만든 데이터 설정 파일
        epochs=100,  # 전체 데이터셋 반복 학습 횟수
        imgsz=640,  # 이미지 크기 (보통 640 사용)
        batch=16,  # 배치 크기 (GPU 메모리에 맞춰 조절, 메모리 부족 시 줄이세요)
        device=0,  # ★ GPU 사용 설정 (0, 1, 2... 또는 [0, 1])
        workers=4,  # 데이터 로딩에 사용할 CPU 프로세스 수 (Windows라면 0 권장)
        project='finger_project',  # 결과가 저장될 프로젝트 폴더 이름
        name='train_result',  # 결과가 저장될 하위 폴더 이름
        exist_ok=True,  # 폴더가 있어도 덮어쓰기 가능 여부
        patience=6,  # 성능 향상이 없으면 10 에포크 뒤 조기 종료
        verbose=True
    )

    print("훈련 완료!")

if __name__ == "__main__":
    main()