import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os

# Define the file paths for the models
# Make sure these paths are correct relative to where you run the script
FACE_CASCADE_PATH = os.path.join('src', 'haarcascade_frontalface_default.xml')
MODEL_WEIGHTS_PATH = 'model.h5' # Assuming model.h5 is in the same directory as the script

# Load the Haar Cascade face detector
# Use try-except to handle potential FileNotFoundError
try:
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
         st.error(f"Error loading face cascade classifier from {FACE_CASCADE_PATH}")
         face_cascade = None
    else:
        st.success("Face cascade classifier loaded successfully.")
except Exception as e:
    st.error(f"Error loading face cascade classifier: {e}")
    face_cascade = None


# Load the pre-trained Keras emotion model weights
# Define the model architecture before loading weights
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1), name='conv2d_1'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv2d_2'))
model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_1'))
model.add(Dropout(0.25, name='dropout_1'))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', name='conv2d_3'))
model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_2'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', name='conv2d_4'))
model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_3'))
model.add(Dropout(0.25, name='dropout_2'))

model.add(Flatten(name='flatten'))
model.add(Dense(1024, activation='relu', name='dense_1'))
model.add(Dropout(0.5, name='dropout_3'))
model.add(Dense(7, activation='softmax', name='dense_2'))


try:
    model.load_weights(MODEL_WEIGHTS_PATH)
    emotion_model = model
    st.success("Emotion model weights loaded successfully.")
except Exception as e:
    emotion_model = None
    st.error(f"Error loading emotion model weights: {e}")
    st.warning("Emotion detection will not be available.")


# Emotion mapping and teacher suggestion function
emotion_mapping = {
    "Angry": "Stressed", "Disgusted": "Bored", "Fearful": "Confused",
    "Happy": "Engaged", "Neutral": "Neutral", "Sad": "Bored", "Surprised": "Confused"
}

def get_teacher_suggestion(student_emotion):
    """
    Provides teacher suggestions based on the student's emotional state.

    Args:
        student_emotion (str): The broader student state derived from emotion mapping.

    Returns:
        str: A suggestion for the teacher.
    """
    if student_emotion == "Confused": return "Suggestion: Re-explain the concept."
    if student_emotion == "Bored": return "Suggestion: Introduce an interactive activity."
    if student_emotion == "Stressed": return "Suggestion: Monitor for consistent stress."
    if student_emotion == "Engaged": return "Student is engaged."
    return "Student is attentive."

class EmotionVideoTransformer(VideoTransformerBase):
    def __init__(self, emotion_model, face_cascade, emotion_mapping, get_teacher_suggestion):
        self.emotion_model = emotion_model
        self.face_cascade = face_cascade
        self.emotion_mapping = emotion_mapping
        self.get_teacher_suggestion = get_teacher_suggestion
        self.emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        processed_frame = img.copy() # Work on a copy to draw

        for (x, y, w, h) in faces:
            # Extract face ROI
            roi_gray = gray_frame[y:y + h, x:x + w]

            # Resize and prepare for model
            cropped_img = cv2.resize(roi_gray, (48, 48))
            cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)
            # Assuming the model expects normalized input if trained with rescale=1./255
            cropped_img = cropped_img / 255.0

            # Predict emotions
            if self.emotion_model: # Check if model is loaded
                prediction = self.emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))

                # Map emotion and get suggestion
                dominant_emotion = self.emotion_dict.get(maxindex, "Unknown")
                student_state = self.emotion_mapping.get(dominant_emotion, "Unknown")
                suggestion = self.get_teacher_suggestion(student_state)

                # Draw rectangle and text
                cv2.rectangle(processed_frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                cv2.putText(processed_frame, dominant_emotion, (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(processed_frame, suggestion, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                # If model is not loaded, just draw the face rectangle
                cv2.rectangle(processed_frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                cv2.putText(processed_frame, "Model not loaded", (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return processed_frame

# Main Streamlit application structure
if __name__ == "__main__":
    st.title("Real-time AI Emotional Analysis for Smart Education")
    st.write("Real-time video stream:")

    if face_cascade and emotion_model:
        webrtc_streamer(
            key="emotion-detection",
            video_processor_factory=lambda: EmotionVideoTransformer(
                emotion_model=emotion_model,
                face_cascade=face_cascade,
                emotion_mapping=emotion_mapping,
                get_teacher_suggestion=get_teacher_suggestion,
            ),
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    else:
        st.error("Models not loaded. Please ensure 'haarcascade_frontalface_default.xml' and 'model.h5' are in the correct paths.")