import cv2
import numpy as np
import mediapipe as mp
import csv
import os

# Create a directory to store the collected data if it doesn't exist
if not os.path.exists('asl_data'):
    os.makedirs('asl_data')

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Set up OpenCV for capturing video from the webcam
cap = cv2.VideoCapture(0)

# Collect data for each sign
signs = ['A', 'B', 'C', 'D', 'E']  # Add more signs as needed
num_samples = 300  # Number of samples to collect per sign

def collect_data_for_sign(sign):
    samples_collected = 0
    all_data = []

    while samples_collected < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB format for MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        # Draw hand landmarks and extract coordinates
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmark data
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                
                # Flatten landmarks and append the sign label
                landmarks_flat = np.array(landmarks).flatten()
                landmarks_flat = np.append(landmarks_flat, signs.index(sign))  # Add label as the last value
                
                # Save the data
                all_data.append(landmarks_flat)
                samples_collected += 1

                # Display progress
                cv2.putText(frame, f"Collecting data for {sign}: {samples_collected}/{num_samples}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the current frame
        cv2.imshow(f"Collecting data for {sign}", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save the data to a CSV file
    with open(f'asl_data/{sign}_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_data)

for sign in signs:
    print(f"Collecting data for {sign}...")
    collect_data_for_sign(sign)
    print(f"Data collection for {sign} complete!")

cap.release()
cv2.destroyAllWindows()
