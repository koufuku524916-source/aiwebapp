import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

model = load_model("keras_model.h5")
class_names = [line.strip() for line in open("labels.txt", "r")]

st.sidebar.title("AI画像認識アプリ")
st.sidebar.write("Teachable Machineの学習モデルを使って画像判定します。")

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

img_source = st.sidebar.radio(
    "画像のソースを選択してください。",
    ("画像をアップロード", "カメラで撮影")
)

if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg"])
else:
    img_file = st.camera_input("カメラで撮影")

if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    st.image(image, caption="対象の画像", width=480)

    image_array = np.asarray(image).astype(np.float32)
    normalized_image_array = (image_array / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)

    st.subheader("円グラフ")
    fig, ax = plt.subplots()
    ax.pie(prediction[0], labels=class_names, autopct="%.2f")
    st.pyplot(fig)

    st.subheader("一覧表")
    st.write(pd.DataFrame(prediction[0], index=class_names, columns=["確率"]))

