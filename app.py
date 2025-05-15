import os
# ── Tame TF logging & threads ───────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
@@ -9,15 +16,15 @@
import numpy as np
from tensorflow.keras.models import load_model

# ——— 1) Load model & labels —————————————————————————————————————
# ── 1) Load model & labels ──────────────────────────────────────────────────
@st.cache_resource
def load_emotion_model():
    return load_model("fer_expression_model.h5")

model = load_emotion_model()
emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# ——— 2) Face detector setup —————————————————————————————————————
# ── 2) Face detector (Haarcascade) ─────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
@@ -28,43 +35,52 @@ def load_emotion_model():

st.title("🎥 Live Emotion Recognition (Webcam Only)")

# ——— 3) Transformer for per-frame processing ———————————————————————
# ── 3) Per‑frame transformer ────────────────────────────────────────────────
class EmotionTransformer(VideoTransformerBase):
    def transform(self, frame):
        # 3a) Get a small frame
        img = frame.to_ndarray(format="bgr24")
        # img is already at requested resolution (we set 320×240 below)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        faces = face_cascade.detectMultiScale(gray, 1.2, 4)

        for (x, y, w, h) in faces:
            # 3b) Crop & resize face
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = cv2.resize(roi, (48,48))
            roi = roi.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=(0, -1))  # shape (1,48,48,1)
            roi = np.expand_dims(roi, axis=(0,-1))  # (1,48,48,1)

            preds = model.predict(roi, verbose=0)[0]
            # 3c) Predict in‐batch (a little faster than predict per sample)
            preds = model.predict_on_batch(roi)[0]
            idx = np.argmax(preds)
            label, conf = emotion_labels[idx], preds[idx]

            # Draw bounding box and label
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # 3d) Draw
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(
                img,
                f"{label} ({conf:.0%})",
                (x, y - 10),
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                (0,255,0),
                2
            )

        return img

# ——— 4) Launch the webcam streamer —————————————————————————————
# ── 4) Launch the streamer with low‐res + frame‐limit ────────────────────────
webrtc_streamer(
    key="emotion-webcam",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_transformer_factory=EmotionTransformer,
    media_stream_constraints={"video": True, "audio": False},
    media_stream_constraints={
        "video": {"width": 320, "height": 240},  # lower res
        "audio": False
    },
    async_transform=True,
    max_frames_in_flight=1  # drop frames if busy
)
