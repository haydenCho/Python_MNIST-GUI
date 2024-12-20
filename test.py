import torch
from model import MNISTModel
from dataset import get_test_dataloader

# 디바이스 설정 (GPU가 사용 가능하면 GPU, 아니면 CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
MODEL_PATH = "model.pth"
model = MNISTModel()  # 사용자 정의 모델 초기화
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))  # 가중치 로드
model = model.to(DEVICE)  # 모델을 디바이스로 이동
model.eval()  # 평가 모드 설정

# 테스트 데이터 로드
test_loader = get_test_dataloader(batch_size=32)

# 테스트 정확도 계산
correct = 0
total = 0
with torch.no_grad():  # 그래디언트 계산 비활성화
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # 모델 추론
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # 가장 높은 확률의 클래스 선택

        # 정확도 계산
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 테스트 정확도 출력
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
