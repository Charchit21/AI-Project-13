import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

EMOTION_LABels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) 
        face_roi = gray[y:y+h, x:x+w]
        
        try:
            resized_face = cv2.resize(face_roi, (48, 48))
            
            blob = cv2.dnn.blobFromImage(
                resized_face, 
                scalefactor=1.0/255.0, 
                size=(48, 48), 
                mean=(0,), 
                swapRB=False, 
                crop=False
            )
            
            dominant_emotion = "Emotion" 

            cv2.putText(
                frame, 
                dominant_emotion, 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                (0, 255, 0), 
                2
            )
        except Exception as e:
            pass

    cv2.imshow('Face and Emotion Detection - Press q to Quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()