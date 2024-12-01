   
    
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 # type: ignore
import numpy as np

# MobileNetV2 modelini yükle
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Model için sınıf etiketleri
class_labels = {948: 'Elma', 966: 'Muz', 906: 'Brokoli', 954: 'Portakal', 943: 'Salatalık', 932: 'Domates', 936: 'Havuç', 937: 'kereviz'}

# Bir görüntüyü işle
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# Görüntüyü sınıflandır
def classify_image(image):
    image = preprocess_image(image)
    predictions = model.predict(image)
    top_prediction = np.argmax(predictions)
    return class_labels.get(top_prediction, 'yiyecek tanınamadı')

# Kameradan görüntü al
def capture_image_from_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılmadı!")
        return None

    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        print("Görüntü yakalanamadı!")
        return None

# Kameradan görüntü al ve sınıflandır
image = capture_image_from_camera()
if image is not None:
    result = classify_image(image)
    print(f"Bu yiyecek: {result}")
else:
    print("Görüntü alınamadı!")

