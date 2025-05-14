import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

# ——— 1) Load model & labels —————————————————————————————————————
@st.cache_resource
def load_emotion_model():
    return load_model("fer_expression_model.h5")

model = load_emotion_model()
emotion_labels = ["angry","disgust","fear","happy","sad","surprise","neutral"]

st.title("📸 Webcam Snapshot Emotion Predictor")

st.write("Click below to take a single webcam photo and predict the facial expression.")

# ——— 2) Camera input —————————————————————————————————————————
img_file = st.camera_input("Take a photo")

if img_file:
    # 3) Display original
    st.image(img_file, caption="📷 Your snapshot", use_column_width=False)
    
    # 4) Preprocess for model
    img = Image.open(img_file).convert("L")            # grayscale
    img = np.array(img)
    img = cv2.resize(img, (48,48))                     # resize to 48×48
    x   = img.astype("float32") / 255.0                # normalize
    x   = np.expand_dims(x, axis=(0,-1))               # shape (1,48,48,1)

    # 5) Predict
    preds = model.predict(x, verbose=0)[0]
    idx   = int(np.argmax(preds))
    label, conf = emotion_labels[idx], preds[idx]

    # 6) Show result
    st.markdown(f"## Prediction: **{label.upper()}**")
    st.markdown(f"Confidence: **{conf:.1%}**")
