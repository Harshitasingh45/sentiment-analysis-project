import cv2
import numpy as np
from tensorflow.keras.models import load_model

# import moodmate assistant
import sys
sys.path.append('./Moodmate')
from moodmate import moodmate_assistant, speak

# Load emotion model
model = load_model('./Moodmate/face_sentiment_model.h5')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load face detector
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start webcam
cap = cv2.VideoCapture(0)

speak("Hello! I am MoodMate. Let me detect your emotion!")

detected_emotion = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        prediction = model.predict(roi)
        detected_emotion = emotion_labels[np.argmax(prediction)]

        cv2.putText(frame, detected_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('MoodMate - Emotion Detection', frame)

    # Press 's' to save emotion and activate assistant
    if cv2.waitKey(1) & 0xFF == ord('s'):
        if detected_emotion:
            print(f"Detected Emotion: {detected_emotion}")
            cap.release()
            cv2.destroyAllWindows()
            # Now run moodmate assistant with detected emotion
            moodmate_assistant(detected_emotion.lower())
            break

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


## ▶️ How It Works

#Run main.py
    
#Webcam opens → detects your face
    
#Shows your emotion on screen in real time
    
#Press 'S' when ready → saves your emotion
    
#MoodMate speaks to you about your emotion
    
#Opens YouTube music based on your mood 🎵