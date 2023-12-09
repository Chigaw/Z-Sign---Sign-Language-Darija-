from anyio import CapacityLimiter
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
st.sidebar.image("Zsign-removebg-preview.png")
st.sidebar.title('Sign Language Detection - Sameer Edlabadkar')
st.sidebar.subheader('-Parameter')




def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):

    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:

        r = height / float(h)
        dim = (int(w * r), height)


    else:

        r = width / float(w)
        dim = (width, int(h * r))


    resized = cv2.resize(image, dim, interpolation=inter)


    return resized

app_mode = st.sidebar.selectbox('Choose the App mode',
['About App','Sign Language to Text','Text to sign Language','Speech to sign Language']
)

if app_mode =='About App':

    t1, t2 = st.columns((0.07,1)) 

    t2.title('Sign Language Detection Using MediaPipe with Streamlit GUI')
    t2.markdown(" tel: 01392 451192 | website: https://www.swast.nhs.uk | email: mailto:data.science@swast.nhs.uk")
        
    
    
    st.markdown('In this application we are using MediaPipe for detecting Sign Language. SpeechRecognition Library of python to recognise the voice and machine learning algorithm which convert speech to the Indian Sign Language .StreamLit is to create the Web Graphical User Interface (GUI) ')
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
    st.markdown('''
              # About Me \n 
                Hey this is Sameer Edlabadkar. Working on the technologies such as **Tensorflow, MediaPipe, OpenCV, ResNet50. \n

                Also check me out on Social Media
                - [YouTube](https://www.youtube.com/@edlabadkarsameer/videos)
                - [LinkedIn](https://www.linkedin.com/in/sameer-edlabadkar-43b48b1a7/)
                - [GitHub](https://github.com/edlabadkarsameer)
              If you are facing any issue while working feel free to mail me on edlabadkarsameer@gmail.com

                ''')
elif app_mode == 'Sign Language to Text':

    # Fonction pour la reconnaissance faciale
    def detect_faces(image):
        # Charger le modèle de détection de visage
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Convertir l'image en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Détecter les visages dans l'image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Dessiner des rectangles autour des visages détectés
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return image

    def main():
        np.set_printoptions(suppress=True)
        model = load_model("keras_Model.h5", compile=False)
        class_names = open("labels.txt", "r").readlines()
        
        st.title("Reconnaissance faciale avec Streamlit")

        # Capture vidéo à partir de la caméra
        cap = cv2.VideoCapture(0)

        # Configuration des paramètres de la caméra
        cap.set(3, 640)  # Largeur de la caméra
        cap.set(4, 480)  # Hauteur de la caméra

        # st.write("Cliquez sur le bouton ci-dessous pour activer la caméra et commencer la reconnaissance faciale.")
        btn_start_camera = st.button("Activer la caméra")


        if btn_start_camera:
            st.warning("La caméra est activée. Pour arrêter, cliquez sur le bouton 'Arrêter la caméra'.")

            # Afficher une image vide pour commencer
            placeholder = st.empty()
            #placeholder.image([], channels="BGR", use_column_width=True, output_format="BGR")

            while True:
                # Capture de la vidéo
                ret, frame = cap.read()

                frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                #cv2.imshow("Webcam Image", frame)
                frame = np.asarray(frame, dtype=np.float32).reshape(1, 224, 224, 3)
                #frame = (frame / 255.0)
                
                
                # Predicts the model
                prediction = model.predict(frame)
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence_score = prediction[0][index]
                time.sleep(1)
                

            # Print prediction and confidence score
                # placeholder.image(frame, channels="BGR", use_column_width=True, output_format="BGR")
                print("Class:", class_name[2:], end="")
                print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
                normalized_frame = frame / 255.0

            # Display camera feed and predictions
                placeholder.image(normalized_frame, channels="BGR", use_column_width=True, output_format="BGR")
                st.write("Class:", class_name[2:])
                st.write("Confidence Score:", f"{np.round(confidence_score * 100)}%")
             

        # Arrêter la capture vidéo lorsque l'utilisateur clique sur "Arrêter la caméra"
        #st.button("Arrêter la caméra")
 

    main()
  
            

            


elif app_mode == 'Text to sign Language':

    st.title('Text to Sign Language (The System use Indian Sign Language)')


    # define function to display sign language images
    def display_images(text):
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


    text = st.text_input("Enter text:")
    # convert text to lowercase
    text = text.lower()

    # display sign language images
    display_images(text)
    
else:
    st.title('Speech to Sign Language (The System use Indian Sign Language)')
    # initialize the speech recognition engine
    # initialize the speech recognition engine
    r = sr.Recognizer()


    # define function to display sign language images
    def display_images(text):
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


    # add start button to start recording audio
    if st.button("Start Talking"):
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

        # convert text to lowercase
        text = text.lower()
        # display the final result
        st.write(f"You said: {text}", font_size=41)

        # display sign language images
        display_images(text)