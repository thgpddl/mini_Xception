import cv2
import numpy as np
import torch
from utils.Model import mini_XCEPTION

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("devicea:",device)

def preprocess_input(x):
    x = x.astype('float32')
    x = x / 255.0
    x = x - 0.5
    x = x * 2.0
    return torch.tensor(x)


src = cv2.imread("test.jpeg")
img=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)

detection_model_path = 'utils/haarcascade_frontalface_default.xml'
emotion_model_path = 'output/E135_0.6466.pth'
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

face_detection = cv2.CascadeClassifier(detection_model_path)
model = mini_XCEPTION(num_classes=7).to(device)
model.load_state_dict(torch.load(emotion_model_path,map_location=device))

input_size = (48, 48)

faces = face_detection.detectMultiScale(img, scaleFactor=1.1, minNeighbors=8)

with torch.no_grad():
    for face_coordinates in faces:
        x, y, w, h = face_coordinates
        gray_face = img[y:y + h, x:x + w]
        try:
            gray_face = cv2.resize(gray_face, input_size)
        except:
            continue
        gray_face = preprocess_input(gray_face)
        inp = torch.unsqueeze(gray_face, 0)
        inp = torch.unsqueeze(inp, 0)
        inp=inp.to(device)
        emotion_label_arg = np.argmax(model(inp)).item()
        emotion_text = emotion_labels[emotion_label_arg]

        print("predict：", emotion_text)
        cv2.rectangle(src, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.imshow("", src)
        cv2.waitKey(0)
