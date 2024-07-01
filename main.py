from yolov10.ultralytics import YOLOv10
import streamlit as st
from PIL import Image


TRAINED_MODEL_PATH = 'best.pt'
model = YOLOv10(TRAINED_MODEL_PATH)

CONF_THRESHOLD = 0.3
IMG_SIZE = 640

st.title("Dectect Helmet using YOLOv10")

file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])
if file is not None:
    st.image(file, caption="Uploaded Image")

    image = Image.open(file)
    results = model.predict(source=image, imgsz=IMG_SIZE, conf=CONF_THRESHOLD)
    annotated_img = results[0].plot()

    st.image(annotated_img, caption="Processed Image")
