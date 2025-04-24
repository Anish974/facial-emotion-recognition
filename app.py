import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model and Haar cascade
model = load_model("model_filter.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Streamlit UI setup
st.set_page_config(page_title="Webcam Emotion Detection", layout="centered")
st.title("🎥 Real-time Facial Emotion Detection")

run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Failed to access webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi, (48, 48))
        roi_normalized = roi_resized.reshape(1, 48, 48, 1) / 255.0

        prediction = model.predict(roi_normalized)
        emotion = emotion_labels[np.argmax(prediction)]

        # Draw bounding box and label
        label_pos = (x, y - 10 if y - 10 > 10 else y + 20)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Show the frame in Streamlit
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

camera.release()
cv2.destroyAllWindows()
