import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    VideoTransformerBase,
    RTCConfiguration,
    WebRtcMode
)
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ——— 1) Load model & labels —————————————————————————————————————
@st.cache_resource
def load_emotion_model():
    return load_model("fer_expression_model.h5")

model = load_emotion_model()
emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# ——— 2) Face detector setup —————————————————————————————————————
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

st.title("🎥 Live Emotion Recognition (Webcam Only)")

# ——— 3) Transformer for per-frame processing ———————————————————————
class EmotionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=(0, -1))  # shape (1,48,48,1)

            preds = model.predict(roi, verbose=0)[0]
            idx = np.argmax(preds)
            label, conf = emotion_labels[idx], preds[idx]

            # Draw bounding box and label
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{label} ({conf:.0%})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        return img

# ——— 4) Launch the webcam streamer —————————————————————————————
webrtc_streamer(
    key="emotion-webcam",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_transformer_factory=EmotionTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)
