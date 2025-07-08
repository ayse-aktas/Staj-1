import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog

class CNNEmotionModel(nn.Module):
    def __init__(self):
        super(CNNEmotionModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = CNNEmotionModel()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

classes = {
    0: "Öfke",
    1: "İğrenme",
    2: "Korku",
    3: "Mutlu",
    4: "Üzgün",
    5: "Şaşkın",
    6: "Nötr"
}

try:
    font_path = "C:/Windows/Fonts/arial.ttf"
    font = ImageFont.truetype(font_path, 24)
except:
    font = ImageFont.load_default()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def analyze_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Görsel yüklenemedi.")
        return

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

    if len(faces) == 0:
        message = "Yüz tespit edilemedi"
    else:
        for (x, y, w, h) in faces:
            face_img = image[y:y+h, x:x+w]
            input_tensor = transform(face_img).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                emotion = classes[predicted.item()]
            draw.rectangle([x, y, x + w, y + h], outline=(85, 255, 0), width=2)
        message = f"Tespit Edilen Duygu: {emotion}"

    img_width, img_height = img_pil.size
    text_area_height = 100
    canvas_height = img_height + text_area_height
    canvas_width = max(img_width, 600)

    canvas = Image.new("RGB", (canvas_width, canvas_height), (30, 30, 30))
    x_offset = (canvas_width - img_width) // 2
    canvas.paste(img_pil, (x_offset, 0))

    draw_canvas = ImageDraw.Draw(canvas)
    bbox = draw_canvas.textbbox((0, 0), message, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (canvas_width - text_width) // 2
    text_y = img_height + (text_area_height - (bbox[3] - bbox[1])) // 2
    draw_canvas.text((text_x, text_y), message, font=font, fill=(255, 255, 255))

    result_bgr = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    cv2.imshow("Görselde Duygu Durumu", result_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def analyze_camera():
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            input_tensor = transform(face_img).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                emotion = classes[predicted.item()]
            draw.rectangle([x, y, x + w, y + h], outline=(85, 255, 0), width=2)
            draw.text((x, y - 30), emotion, font=font, fill=(255, 255, 255, 0))

        result_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        cv2.imshow("Kamera ile Yüz İfadesi", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

while True:
    print("\n--- Yüz İfadesi Tanıma ---")
    print("1. Kamera ile gerçek zamanlı")
    print("2. Görsel dosya ile analiz")
    print("q. Çıkış")

    secim = input("Seçiminizi yapınız (1 / 2 / q): ").strip().lower()

    if secim == "1":
        analyze_camera()
    elif secim == "2":
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("Görsel Dosyaları", "*.jpg *.jpeg *.png")])
        if file_path:
            analyze_image(file_path)
        else:
            print("Dosya seçilmedi.")
    elif secim == "q":
        print("Programdan çıkılıyor.")
        break
    else:
        print("Geçersiz seçim. Lütfen 1, 2 veya q girin.")
