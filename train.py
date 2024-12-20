import torch
import torch.nn as nn
import torch.optim as optim
from model import MNISTModel
from dataset import get_mnist_dataloaders

# 하이퍼파라미터
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 데이터 로드
train_loader, val_loader = get_mnist_dataloaders(BATCH_SIZE)

# 모델, 손실 함수, 옵티마이저 초기화
model = MNISTModel().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 학습 루프
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # 순전파
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 역전파 및 가중치 갱신
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 검증
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}, Validation Accuracy: {100 * correct / total:.2f}%")

# 모델 저장
torch.save(model.state_dict(), 'model.pth')
