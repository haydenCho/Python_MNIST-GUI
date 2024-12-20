import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split

# 데이터셋 로드 함수
def get_mnist_dataloaders(batch_size=64, validation_split=0.2):
    # MNIST 데이터셋 로드 및 전처리
    transform = transforms.Compose([
        transforms.ToTensor(),                # 텐서로 변환
        transforms.Normalize((0.5,), (0.5,)) # 평균 0.5, 표준편차 0.5로 정규화
    ])
    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # 데이터셋 분할 (훈련/검증)
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# 테스트 데이터 로드 함수
def get_test_dataloader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader
