import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('asl_recognition_model.h5')

# ASL sign labels (modify this list to match the signs you used for training)
signs = ['A', 'B', 'C', 'D', 'E']  # Add more labels as necessary

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Set up OpenCV for capturing video from the webcam
cap = cv2.VideoCapture(0)

# Function to process the frame and make predictions
def predict_sign(frame, model):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract hand landmark data
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            # Convert landmarks to a NumPy array and reshape for model input
            landmarks_array = np.array(landmarks).flatten().reshape(1, -1)

            # Predict the sign using the pre-trained model
            prediction = model.predict(landmarks_array)
            predicted_sign = np.argmax(prediction)
            predicted_label = signs[predicted_sign]

            # Display the predicted sign on the frame
            cv2.putText(frame, f"Sign: {predicted_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

# Run real-time sign recognition
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Call the prediction function
    frame = predict_sign(frame, model)

    # Display the frame
    cv2.imshow('ASL Sign Language Recognition', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
