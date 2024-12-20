import torch
from torchvision import transforms
from PIL import Image

def predict_image(image_path, model, device):
    # 1. 이미지 로드 및 전처리
    transform = transforms.Compose([
        transforms.Grayscale(),               # MNIST는 흑백 이미지
        transforms.Resize((28, 28)),          # MNIST 이미지는 28x28
        transforms.ToTensor(),                # 텐서로 변환
        transforms.Normalize((0.5,), (0.5,))  # 정규화
    ])
    image = Image.open(image_path).convert('L')  # 흑백으로 변환
    image = transform(image).unsqueeze(0)       # 배치 차원 추가
    image = image.to(device)                    # 데이터 텐서를 GPU로 이동

    # 2. 모델 추론
    model.eval()                                # 모델을 평가 모드로 전환
    with torch.no_grad():                       # 그래디언트 계산 비활성화
        output = model(image)
        _, predicted = torch.max(output, 1)     # 가장 높은 확률의 클래스 선택

    return predicted.item()                     # 예측된 클래스 반환
