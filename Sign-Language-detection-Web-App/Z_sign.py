import pickle
import threading
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import os
import speech_recognition as sr

from keras.models import load_model  # TensorFlow is required for Keras to work

st.set_page_config(page_title="Zsign",page_icon='ZsignLogo-removebg-preview.png',layout="wide")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils


DEMO_VIDEO = 'demo.mp4'
DEMO_IMAGE = 'demo.jpg'

my_list = [] 



st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
def resiezeImage(img,new_width,new_height) :
    original_image = Image.open(img)
    # Obtenir les dimensions originales de l'image
     # Redimensionner l'image
    resized_image = original_image.resize((new_width, new_height))
    return resized_image
           



logo='ZsignLogo-removebg-preview.png'
st.sidebar.image(resiezeImage(logo,300,250))
st.sidebar.title('Sign Language Detection - English - Darija Morocco ')



app_mode = st.sidebar.selectbox('Choose the App mode',
['About App','Sign Language to Text','Text to sign Language','Speech to sign Language','write words using sign language','Video To Sign Language']
)

if app_mode =='About App':

    t1, t2 = st.columns((0.4,1)) 
    t1.image(resiezeImage(logo,200,200))
    t2.title('Zsign : Sign Language Detection - English - Darija Morocco ')
        
    
    
  
    st.markdown('To facilitate more inclusive and barrier-free communication between hearing and deaf individuals, contributing to a more equitable and understanding society.')


    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    
    aboutAs='about.png'
    t3, t4 = st.columns((0.2,1)) 
    t3.image(resiezeImage(aboutAs,100,100))
    t4.markdown('''# About As \n ''')
    st.markdown(''' 
                The "Zsign" application is an interactive platform designed to facilitate communication between hearing and deaf individuals by interpreting gestures and signs in English and Darija(MOROCCO) Sign Language . Equipped with intelligent image processing and gesture recognition features, the application enables smooth and natural communication. 
                Also check me out on Social Media
                ''')
    t5, t6 = st.columns((0.2,1)) 
    keyFertures='keyfetures.png'
    t5.image(resiezeImage(keyFertures,80,80))
    t6.markdown('''# Key Features: \n ''')
    st.markdown('''
                - An intuitive user interface with visual instructions to guide users.
                - Uses a camera to detect gestures and signs made by the user.
                - Incorporates sign recognition algorithms to interpret gestures in  Sign Language.
                - Converts detected signs into displayed text and/or vocal output for bidirectional communication.
                
                ''')
    t7, t8 = st.columns((0.2,1)) 
    target='target.png'
    t7.image(target,100,100)
    t8.markdown('''# Target Users \n ''')
    st.markdown('''
                - Deaf or hard-of-hearing individuals.
                - Friends, family members, or colleagues of deaf or hard-of-hearing individuals.
                ''')
elif app_mode == 'Sign Language to Text':
    t9, t10 = st.columns((0.4,1)) 
    t9.image(resiezeImage(logo,200,200))
    t10.title("Sign Language to Text")
    st.header('Choose the language')
    langage_mode=st.selectbox("Pick one", ["Darija Morroco", "English"])
    
   
    def AEnglishSign():
        st.title('English Sign Language')
        np.set_printoptions(suppress=True)
        model = load_model("keras_Model.h5", compile=False)
        class_names = open("labels.txt", "r").readlines()

        # Capture vidéo à partir de la caméra
        cap = cv2.VideoCapture(0)

        # Configuration des paramètres de la caméra
        cap.set(3, 320)  # Set width to 320
        cap.set(4, 240)  # Hauteur de la caméra

        st.subheader('Click the button below to activate the camera and start facial recognition.')
        btn_start_camera = st.button("Activate Camera")

        if btn_start_camera:
            st.warning("The camera is activated. To stop, click on the 'Stop Camera' button.")

            # Placeholder to display the webcam feed
            placeholder = st.empty()

            while True:
                # Capture de la vidéo
                ret, frame = cap.read()

                # Resize and preprocess the frame
                frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                frame = np.asarray(frame, dtype=np.float32).reshape(1, 224, 224, 3)
                frame = (frame / 255.0) 

                # Predict the model
                prediction = model.predict(frame)
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence_score = prediction[0][index]
                previous_prediction = class_name

                # Check if the current prediction is different from the previous one
                if class_name != previous_prediction:
                    # Display the webcam feed
                    #placeholder.image(frame, channels="BGR", use_column_width=True, output_format="BGR")
                    placeholder.image(frame, channels="BGR", use_container_width=True, output_format="BGR")

                    # Print prediction and confidence score
                    st.write(class_name[2:], end="")
                    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

                    # Update the previous prediction

                # Sleep for a short time to control the frame rate
                time.sleep(0.1)
        # Arrêter la capture vidéo lorsque l'utilisateur clique sur "Arrêter la caméra"
        if st.button("Arrêter la caméra"):
            cap.release()
    def EnglishSign():
        # st.title('Arabic alphabet')
        # st.subheader('arabic sign language alphabet')
        # st.image('arabicHorof.png')
        st.title('English alphabet ')
        st.subheader('English sign language alphabet')
        # st.image('1-1.jpg')
        np.set_printoptions(suppress=True)
        model_dict = pickle.load(open('C:\\Users\\DELL\\Desktop\\Z-Sign---Sign-Language-Darija-\\Sign-Language-detection-Web-App\\data_words.pickle', 'rb'))
        model = model_dict['model']

        cap = cv2.VideoCapture(0)
        cap.set(3, 320)  # Set width to 320
        cap.set(4, 240) 

        st.subheader('Click the button below to activate the camera and start facial recognition.')
        btn_start_camera = st.button("Activer la caméra")

        # Initialize MediaPipe Hands
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        # Dictionary to map class indices to labels
        labels_dict = {0: 'hello', 1: 'our goal', 2: 'is to help', 3: 'deaf people'}

        if btn_start_camera:
            st.warning("La caméra est activée. Pour arrêter, cliquez sur le bouton 'Arrêter la caméra'.")
            current_word = ""
            cooldown_start_time = time.time()
            cooldown_duration = 2.0  # Set the cooldown duration in seconds
            placeholder = st.empty()

            while btn_start_camera:  # Keep running as long as the camera is activated
                ret, frame = cap.read()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    data_aux = []
                    x_ = []
                    y_ = []
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

                        # Display the prediction on the Streamlit page
                        st.write(f"Prediction: {predicted_character}")
                       
                        # Update the current word
                        current_word += predicted_character

                        # Reset cooldown timer
                        cooldown_start_time = time.time()

                    placeholder.image(frame, channels="BGR", use_column_width=True, output_format="BGR")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, current_word, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                                cv2.LINE_AA)

            # Arrêter la capture vidéo lorsque l'utilisateur clique sur "Arrêter la caméra"
        if st.button("Arrêter la caméra"):
            cap.release()

    def DarijaSign():
          # st.title('Arabic alphabet')
                # st.subheader('arabic sign language alphabet')
                # st.image('arabicHorof.png')
                # st.title('English alphabet ')
                st.subheader('English sign language')
                # st.image('1-1.jpg')
                np.set_printoptions(suppress=True)
                model_dict = pickle.load(open('C:\\Users\\DELL\\Desktop\\Z-Sign---Sign-Language-Darija-\\Sign-Language-detection-Web-App\\data_words.pickle', 'rb'))
                model = model_dict['model']

                cap = cv2.VideoCapture(0)
                cap.set(3, 320)  # Set width to 320
                cap.set(4, 240) 

                st.subheader('Click the button below to activate the camera and start facial recognition.')
                btn_start_camera = st.button("Activer la caméra")

                # Initialize MediaPipe Hands
                mp_hands = mp.solutions.hands
                mp_drawing = mp.solutions.drawing_utils
                hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

                # Dictionary to map class indices to labels
                labels_dict = {0: 'سلام', 1: 'الهدف ديالنا', 2: 'نعاونو', 3: 'الناس لمكتسمعش'}

                if btn_start_camera:
                    st.warning("La caméra est activée. Pour arrêter, cliquez sur le bouton 'Arrêter la caméra'.")
                    current_word = ""
                    cooldown_start_time = time.time()
                    cooldown_duration = 2.0  # Set the cooldown duration in seconds
                    placeholder = st.empty()

                    while btn_start_camera:  # Keep running as long as the camera is activated
                        ret, frame = cap.read()
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = hands.process(frame_rgb)

                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                mp_drawing.draw_landmarks(
                                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                            data_aux = []
                            x_ = []
                            y_ = []
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

                                # Display the prediction on the Streamlit page
                                st.write(f"Prediction: {predicted_character}")
                            
                                # Update the current word
                                current_word += predicted_character

                                # Reset cooldown timer
                                cooldown_start_time = time.time()

                            placeholder.image(frame, channels="BGR", use_column_width=True, output_format="BGR")
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                            cv2.putText(frame, current_word, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                                        cv2.LINE_AA)

                    # Arrêter la capture vidéo lorsque l'utilisateur clique sur "Arrêter la caméra"
                if st.button("Arrêter la caméra"):
                    cap.release()          

            # Arrêter la capture vidéo lorsque l'utilisateur clique sur "Arrêter la caméra"
        # if st.button("Arrêter la caméra"):
        #     cap.release()

    if langage_mode=='Darija Morroco' : 
        DarijaSign()
       
    elif langage_mode=='English':
        EnglishSign()

            


elif app_mode == 'Text to sign Language':

    st.title('Text to Sign Language ')
    st.header('Converting text to sign language involves translating written text into corresponding signs or gestures in a sign language system')
    st.subheader('Choose the language')
    txt_mode=st.selectbox("Pick one", ["Darija Morroco", "English"])

    # define function to display sign language images
    def EnglishText():
        # get the file path of the images directory
        st.title('English text to English Sign language')
        img_dir = "photo/"
        text = st.text_input("Enter text:")
        text = text.lower()
        # initialize variable to track image position
        image_pos = st.empty()

        # iterate through the text and display sign language images
        for char in text:
            if char.isalpha():
                # display sign language image for the alphabet
                img_path = os.path.join(img_dir, f"{char}.jpeg")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=500)

                # wait for 2 seconds before displaying the next image
                time.sleep(1)

                # remove the image
                image_pos.empty()
            elif char == ' ':
                # display space image for space character
                img_path = os.path.join(img_dir, "space.png")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=500)

                # wait for 2 seconds before displaying the next image
                time.sleep(1)

                # remove the image
                image_pos.empty()

        # wait for 2 seconds before removing the last image
        time.sleep(2)
        image_pos.empty()


    def DarijaText():
        st.title('Darija text to Darija Sign language')
        # get the file path of the images directory
        img_dir = "photoArab/"
        text = st.text_input("Enter text:")
        text = text.lower()
        # initialize variable to track image position
        image_pos = st.empty()

        # iterate through the text and display sign language images
        for char in text:
            if char.isalpha():
                # display sign language image for the alphabet
                img_path = os.path.join(img_dir, f"{char}.JPG")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=500)

                # wait for 2 seconds before displaying the next image
                time.sleep(1)

                # remove the image
                image_pos.empty()
            elif char == ' ':
                # display space image for space character
                img_path = os.path.join(img_dir, "space.png")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=500)

                # wait for 2 seconds before displaying the next image
                time.sleep(1)

                # remove the image
                image_pos.empty()

        # wait for 2 seconds before removing the last image
        time.sleep(2)
        image_pos.empty()

    
    # convert text to lowercase
    if txt_mode=='English' :
        # display sign language images
        EnglishText()
    else:
        DarijaText()
        
elif app_mode == 'Speech to sign Language':
    st.title('Speech to Sign Language ')
    # initialize the speech recognition engine
    # initialize the speech recognition engine
    r = sr.Recognizer()
    st.header('Converting a speech into sign language involves translating spoken words and expressions into visual gestures and movements. ')
    st.subheader('Choose the language')
    txt_mode=st.selectbox("Pick one", ["Darija Morroco", "English"])

    # define function to display sign language images
    def display_English_images(text):
        # get the file path of the images directory
        img_dir = "photo/"

        # initialize variable to track image position
        image_pos = st.empty()

        # iterate through the text and display sign language images
        for char in text:
            if char.isalpha():
                # display sign language image for the alphabet
                img_path = os.path.join(img_dir, f"{char}.jpeg")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=300)

                # wait for 2 seconds before displaying the next image
                time.sleep(2)

                # remove the image
                image_pos.empty()
            elif char == ' ':
                # display space image for space character
                img_path = os.path.join(img_dir, "space.png")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=300)

                # wait for 2 seconds before displaying the next image
                time.sleep(2)

                # remove the image
                image_pos.empty()

        # wait for 2 seconds before removing the last image
        time.sleep(2)
        image_pos.empty()

        # define function to display sign language images
    def display_Arabic_images(text):
    # get the file path of the images directory
        img_dir = "photoArab/"

        # initialize variable to track image position
        image_pos = st.empty()

        # iterate through the text and display images for Arabic characters
        for char in text:
            if char.isalpha():
                # display image for the Arabic alphabet
                img_path = os.path.join(img_dir, f"{char}.JPG")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=300)

                # wait for 2 seconds before displaying the next image
                time.sleep(1.5)

                # remove the image
                image_pos.empty()
            elif char == ' ':
                # display space image for space character
                img_path = os.path.join(img_dir, "space.png")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=300)

                # wait for 2 seconds before displaying the next image
                time.sleep(2)

                # remove the image
                image_pos.empty()

        # wait for 2 seconds before removing the last image
        time.sleep(2)
        image_pos.empty()

    # add start button to start recording audio
    if st.button("Start Talking"):
        if txt_mode=='English':
        # record audio for 5 seconds
            with sr.Microphone() as source:
                st.write("Say something!")
                audio = r.listen(source, phrase_time_limit=5)

                try:
                    text = r.recognize_google(audio)
                except sr.UnknownValueError:
                    st.write("Sorry, I did not understand what you said.")
                except sr.RequestError as e:
                    st.write(f"Could not request results from Google Speech Recognition service; {e}")
            st.write(f"You said: {text}", font_size=41)
       
            text = text.lower()
            # display sign language images
            display_English_images(text)

        else:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                st.write("Say something!")
                audio = r.listen(source, phrase_time_limit=5)

                try:
                    # transcribe the speech to text using Google Speech Recognition
                    text = r.recognize_google(audio, language='ar')
                except sr.UnknownValueError:
                    st.write("Sorry, I did not understand what you said.")
                except sr.RequestError as e:
                    st.write(f"Could not request results from Google Speech Recognition service; {e}")

            # convert text to lowercase
            text = text.lower()

            # display the final result
            st.write(f"You said: {text}", font_size=41)

            # display sign language images for Arabic text
            display_Arabic_images(text)
elif app_mode=='write words using sign language' : 
    
    st.title("Choose your language")
    txt_mode_alph=st.selectbox("Pick one", ["Darija Morroco", "English"])

    def horofenglishSign():
        # st.title('Arabic alphabet')
        # st.subheader('arabic sign language alphabet')
        # st.image('arabicHorof.png')
        st.title('English alphabet ')
        st.subheader('English sign language alphabet')
        st.image('1-1.jpg')
        np.set_printoptions(suppress=True)
        model_dict = pickle.load(open('C:\\Users\\DELL\\Desktop\\Z-Sign---Sign-Language-Darija-\\Sign-Language-detection-Web-App\\data_anglais.pickle', 'rb'))
        model = model_dict['model']

        cap = cv2.VideoCapture(0)
        cap.set(3, 320)  # Set width to 320
        cap.set(4, 240) 

        st.subheader('Click the button below to activate the camera and start facial recognition.')
        btn_start_camera = st.button("Activer la caméra")

        # Initialize MediaPipe Hands
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        # Dictionary to map class indices to labels
        labels_dict = {0: 'H', 1: 'E', 2: 'L', 3: 'L', 4: 'O'}

        if btn_start_camera:
            st.warning("La caméra est activée. Pour arrêter, cliquez sur le bouton 'Arrêter la caméra'.")
            current_word = ""
            cooldown_start_time = time.time()
            cooldown_duration = 2.0  # Set the cooldown duration in seconds
            placeholder = st.empty()

            while btn_start_camera:  # Keep running as long as the camera is activated
                ret, frame = cap.read()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    data_aux = []
                    x_ = []
                    y_ = []
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
                    expected_features = 42
                    data_aux = data_aux[:expected_features] 

                    x1 = int(min(x_) * frame.shape[1]) - 10
                    y1 = int(min(y_) * frame.shape[0]) - 10
                    x2 = int(max(x_) * frame.shape[1]) - 10
                    y2 = int(max(y_) * frame.shape[0]) - 10

                    # Check if cooldown period has passed
                    if time.time() - cooldown_start_time >= cooldown_duration:
                        prediction = model.predict([np.asarray(data_aux)])
                        predicted_character = labels_dict[int(prediction[0])]

                        # Display the prediction on the Streamlit page
                        st.write(f"Prediction: {predicted_character}")
                       
                        # Update the current word
                        current_word += predicted_character

                        # Reset cooldown timer
                        cooldown_start_time = time.time()

                    placeholder.image(frame, channels="BGR", use_column_width=True, output_format="BGR")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, current_word, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                                cv2.LINE_AA)

            # Arrêter la capture vidéo lorsque l'utilisateur clique sur "Arrêter la caméra"
        if st.button("Arrêter la caméra"):
            cap.release()


    def horofarabicSign():
            st.title('Arabic alphabet')
            st.subheader('arabic sign language alphabet')
            st.image('arabicHorof1.jpg')
            np.set_printoptions(suppress=True)
            model_dict = pickle.load(open('C:\\Users\\DELL\\Desktop\\Z-Sign---Sign-Language-Darija-\\Sign-Language-detection-Web-App\\data_arab.pickle', 'rb'))
            model = model_dict['model']

            cap = cv2.VideoCapture(0)
            cap.set(3, 320)  # Set width to 320
            cap.set(4, 240) 

            st.subheader('Click the button below to activate the camera and start facial recognition.')
            btn_start_camera = st.button("Activer la caméra")

            # Initialize MediaPipe Hands
            mp_hands = mp.solutions.hands
            mp_drawing = mp.solutions.drawing_utils
            hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

            # Dictionary to map class indices to labels
            labels_dict = {0: 'ا', 1: 'ب', 2: 'ت', 3: 'ث', 4: 'ج', 5: 'ح', 6: 'خ', 7: 'د', 8: 'ذ', 9: 'ر'}

            if btn_start_camera:
                st.warning("La caméra est activée. Pour arrêter, cliquez sur le bouton 'Arrêter la caméra'.")
                current_word = ""
                cooldown_start_time = time.time()
                cooldown_duration = 2.0  # Set the cooldown duration in seconds
                placeholder = st.empty()

                while btn_start_camera:  # Keep running as long as the camera is activated
                    ret, frame = cap.read()
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame_rgb)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        data_aux = []
                        x_ = []
                        y_ = []
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

                            # Display the prediction on the Streamlit page
                            st.write(f"Prediction: {predicted_character}")

                            # Update the current word
                            current_word += predicted_character

                            # Reset cooldown timer
                            cooldown_start_time = time.time()

                        placeholder.image(frame, channels="BGR", use_column_width=True, output_format="BGR")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                        cv2.putText(frame, current_word, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                                    cv2.LINE_AA)

                # Arrêter la capture vidéo lorsque l'utilisateur clique sur "Arrêter la caméra"
            if st.button("Arrêter la caméra"):
                cap.release()
    
    
    
     # convert text to lowercase
    if txt_mode_alph=='English' :
        # display sign language images
        horofenglishSign()
    else:
        horofarabicSign()

else : 
    st.title('Video To Sign Language')
    model = load_model("keras_Model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    video_file_buffer = st.file_uploader("Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])

    if video_file_buffer is not None:
        tfflie = tempfile.NamedTemporaryFile(delete=False)
        tfflie.write(video_file_buffer.read())
        cap = cv2.VideoCapture(tfflie.name)
        cap.set(3, 640)  # Largeur de la caméra
        cap.set(4, 480)  # Hauteur de la caméra
        placeholder = st.empty()

        while True:
            # Capture de la vidéo
            ret, frame = cap.read()

            # Resize and preprocess the frame
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            frame = np.asarray(frame, dtype=np.float32).reshape(1, 224, 224, 3)
            frame = (frame / 255.0)

            # Predict the model
            prediction = model.predict(frame)
            index = np.argmax(prediction)
            class_name = class_names[index]

            # Display the webcam feed
            placeholder.image(frame, channels="BGR", use_column_width=True, output_format="BGR")

            # Print prediction
            st.write(class_name[2:], end=" ")

            # Sleep for a short time to control the frame rate
            time.sleep(0.1)

            # Arrêter la capture vidéo lorsque l'utilisateur clique sur "Arrêter la caméra"
    else:
        st.warning("Please upload a video file.")