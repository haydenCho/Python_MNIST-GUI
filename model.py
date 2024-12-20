import torch.nn as nn

# CNN 모델 정의
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 입력 채널=1 (흑백), 출력 채널=32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 크기 절반 축소
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 출력 채널=64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)        # 크기 절반 축소
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),                                # 2D -> 1D 변환
            nn.Linear(64 * 7 * 7, 128),                 # FC 레이어 1
            nn.ReLU(),
            nn.Linear(128, 10)                          # FC 레이어 2 (출력: 10 클래스)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
