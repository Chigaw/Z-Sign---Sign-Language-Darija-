import pickle
import cv2
import mediapipe as mp
import numpy as np

# Charger le modèle depuis le fichier pickle
model_dict = pickle.load(open('/Users/pc/hehehe/data.pickle', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

# Initialiser MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionnaire pour mapper les indices de classe à des étiquettes
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'}

while True:
    # Initialiser des variables pour stocker les coordonnées normalisées des landmarks
    data_aux = []
    x_ = []
    y_ = []

    # Lire le frame de la webcam
    ret, frame = cap.read()

    # Convertir le frame en RGB pour MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processus de détection des landmarks des mains
    results = hands.process(frame_rgb)

    # Traitement des résultats
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Collecte des coordonnées normalisées des landmarks
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

        # Assurez-vous que data_aux a la bonne taille (84 dans ce cas)
        data_aux = data_aux + [0] * (84 - len(data_aux))

        # Obtenir les coordonnées pour la boîte englobante
        x1 = int(min(x_) * frame.shape[1]) - 10
        y1 = int(min(y_) * frame.shape[0]) - 10
        x2 = int(max(x_) * frame.shape[1]) - 10
        y2 = int(max(y_) * frame.shape[0]) - 10

        # Prédiction de la classe
        prediction = model.predict([np.asarray(data_aux)])

        # Diagnostic des prédictions pendant le test
        predicted_character = labels_dict[int(prediction[0])]
        print("Test - Predicted label:", predicted_character)

        # Affichage de la boîte englobante et de la prédiction
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Affichage du frame
    cv2.imshow('frame', frame)

    # Attendre une touche et fermer la fenêtre si la touche 'q' est enfoncée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libération de la capture vidéo et fermeture de la fenêtre
cap.release()
cv2.destroyAllWindows()
