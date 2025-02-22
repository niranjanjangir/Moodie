import cv2
import numpy as np

import pygame
import time
import streamlit as st
from PIL import Image
from deepface import DeepFace



pygame.mixer.init()
alert_sound = "alert.wav"  # Change to your alert file

def detect_emotion():
    cap = cv2.VideoCapture(0)
    distracted_time = 0
    alert_triggered = False

    stframe = st.empty()  # Streamlit UI placeholder

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert OpenCV image to PIL for Streamlit display
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)

        # Analyze emotion
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']

            # Display emotion on UI
            stframe.image(img_pil, caption=f"Detected Emotion: {emotion}", use_column_width=True)

            # If emotion is "tired" or "sad", track distraction time
            if emotion in ["tired", "sad", "neutral"]:  
                if not alert_triggered:
                    distracted_time = time.time()
                    alert_triggered = True  
                elif time.time() - distracted_time > 60:  # If 1 minute distracted
                    pygame.mixer.Sound(alert_sound).play()
                    st.warning("🚨 ALERT: Stay Focused!")
                    alert_triggered = False  # Reset timer

            else:
                alert_triggered = False  # Reset if focus returns

        except Exception as e:
            st.error(f"Error: {e}")

        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


st.title("🧠 AI Mood Tracker")
st.write("This app detects your emotions in real-time and alerts you if you're distracted for too long.")

# Add a button to start detection
if st.button("Start Tracking"):
    detect_emotion()
