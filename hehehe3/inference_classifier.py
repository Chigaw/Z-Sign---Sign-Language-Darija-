import time
import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the model from the pickle file
model_dict = pickle.load(open('/Users/pc/hehehe3/data.pickle', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary to map class indices to labels
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'}

current_word = ""
cooldown_start_time = time.time()
cooldown_duration = 2.0  # Set the cooldown duration in seconds

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        data_aux = data_aux + [0] * (84 - len(data_aux))

        x1 = int(min(x_) * frame.shape[1]) - 10
        y1 = int(min(y_) * frame.shape[0]) - 10
        x2 = int(max(x_) * frame.shape[1]) - 10
        y2 = int(max(y_) * frame.shape[0]) - 10

        # Check if cooldown period has passed
        if time.time() - cooldown_start_time >= cooldown_duration:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Update the current word
            current_word += predicted_character

            # Reset cooldown timer
            cooldown_start_time = time.time()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, current_word, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
