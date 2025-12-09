import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os


# ==========================================
# 1. 모델 정의 (Batch Normalization 추가)
# ==========================================
class TouchClassifier(nn.Module):
    def __init__(self):
        super(TouchClassifier, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # [추가] 학습 안정화
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # [추가]
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # [추가]
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),  # 뉴런 수 약간 증가
            nn.ReLU(),
            nn.Dropout(0.5),  # 과대적합 방지
            nn.Linear(256, 2)  # 0: Hover, 1: Touch
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ==========================================
# 2. 학습 설정 및 실행
# ==========================================
def train():
    DATA_DIR = "touch_dataset"
    if not os.path.exists(DATA_DIR):
        print("❌ 데이터셋 폴더가 없습니다.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 장치: {device}")

    # ★ [핵심] 데이터 증강 (Augmentation) ★
    # 학습용 데이터에는 변형을 주어 강하게 키웁니다.
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),  # 좌우 반전
        transforms.RandomRotation(10),  # 약간 회전 (-10~10도)
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 밝기/대비 변화
        transforms.ToTensor(),
    ])

    # 검증용 데이터는 변형 없이 원본 그대로 평가합니다.
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # 전체 데이터셋 로드
    full_dataset = datasets.ImageFolder(DATA_DIR)  # transform은 나중에 적용

    # 학습(80%) / 검증(20%) 분리
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])

    # transform 적용을 위해 래퍼(Wrapper) 사용 또는 데이터셋 분리 로직 수정 필요
    # 간단하게 구현하기 위해 여기서는 ImageFolder를 두 번 불러서 split index로 나눕니다.
    # (실무에서는 Custom Dataset 클래스를 씁니다)
    train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)

    # 인덱스로 서브셋 생성
    train_subset = torch.utils.data.Subset(train_dataset, train_data.indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_data.indices)

    # 데이터 로더 (Batch Size: 32 추천)
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    print(f"데이터 개수: 학습 {len(train_subset)}장 / 검증 {len(val_subset)}장")

    # 모델 & 설정
    model = TouchClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 초기 학습률

    # 학습률 스케줄러 (성능 정체 시 학습률 감소)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # 학습 루프
    EPOCHS = 50  # 에포크 늘림
    best_loss = float('inf')

    print("=== 학습 시작 ===")

    for epoch in range(EPOCHS):
        # [훈련 모드]
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # [검증 모드]
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        # 스케줄러 업데이트
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        # 최고 성능 모델 저장 (검증 손실 기준)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "touch_classifier_best.pth")
            print("  --> 💾 최고 모델 저장됨!")

    print("=== 학습 완료 ===")
    print(f"최종 모델은 'touch_classifier_best.pth'로 저장되었습니다.")


if __name__ == "__main__":
    train()