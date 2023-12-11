import os
import pickle
import numpy as np
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = '/Users/pc/withH5model/data'

data = []
labels = []
max_landmarks = 21  # Set a fixed number of landmarks

for dir_ in os.listdir(DATA_DIR):
    subdir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(subdir_path):
        for img_path in os.listdir(subdir_path):
            data_aux = []

            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(subdir_path, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                # Normalize and append landmarks
                for i in range(max_landmarks):
                    if i < len(hand_landmarks.landmark):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))
                    else:
                        data_aux.extend([0, 0])

                data.append(data_aux)
                labels.append(dir_)

# Save the data to a pickle file
with open('/Users/pc/withH5model/data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
