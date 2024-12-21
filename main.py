import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from model import MNISTModel  # 모델 클래스스
from predict import predict_image  # 모델 사용(예측) 함수

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
MODEL_PATH = "model.pth"
model = MNISTModel()  # 모델 초기화
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))  # 가중치 로드
model = model.to(DEVICE)  # 모델을 디바이스로 이동
model.eval()  # 평가 모드 설정


# 이미지 선택 및 예측 함수
def open_and_predict():
    # 파일 선택 다이얼로그
    file_path = filedialog.askopenfilename(
        title="사진 고르기",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    if not file_path:
        return  # 파일 선택 취소 시 종료

    # PIL을 사용해 이미지 로드 및 디스플레이
    img = Image.open(file_path)
    img_resized = img.resize((200, 200))  # GUI에 맞게 이미지 크기 조정
    img_tk = ImageTk.PhotoImage(img_resized)
    img_label.config(image=img_tk)
    img_label.image = img_tk

    # 버튼 텍스트 변경
    select_button.config(text="다른 사진 고르기")

    # 예측 수행
    try:
        predicted_number = predict_image(file_path, model, DEVICE)  # 예측 함수 호출
        result_label.config(text=f"이미지의 숫자는? {predicted_number}")
    except Exception as e:
        result_label.config(text=f"Error: {str(e)}")


# GUI 생성
root = tk.Tk()
root.title("MNIST 숫자 분류")
root.geometry("400x400")

# 위젯 구성
title_label = tk.Label(root, text="MNIST 숫자 분류", font=("Arial", 16))
title_label.pack(pady=10)

img_label = tk.Label(root)  # 이미지가 표시될 공간
img_label.pack(pady=10)

result_label = tk.Label(root, text="이미지의 숫자는?", font=("Arial", 14))
result_label.pack(pady=10)

# 버튼들을 담을 프레임 생성
button_frame = tk.Frame(root)
button_frame.pack(pady=20)

# 사진 고르기 버튼
select_button = tk.Button(button_frame, text="사진 고르기", command=open_and_predict)
select_button.pack(side="left", padx=10)

# 끝내기 버튼
exit_button = tk.Button(button_frame, text="끝내기", command=root.destroy)
exit_button.pack(side="left", padx=10)

# GUI 실행
root.mainloop()