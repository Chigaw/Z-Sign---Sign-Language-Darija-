import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import os
import speech_recognition as sr

from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math

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

st.sidebar.title('Sign Language Detection - Sameer Edlabadkar')
st.sidebar.subheader('-Parameter')

@st.cache()
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
['About App','Sign Language to Text','Speech to sign Language']
)

if app_mode =='About App':
    st.title('Sign Language Detection Using MediaPipe with Streamlit GUI')
    st.markdown('In this application we are using **MediaPipe** for detecting Sign Language. **SpeechRecognition** Library of python to recognise the voice and machine learning algorithm which convert speech to the Indian Sign Language .**StreamLit** is to create the Web Graphical User Interface (GUI) ')
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
                Hey this is **Sameer Edlabadkar**. Working on the technologies such as **Tensorflow, MediaPipe, OpenCV, ResNet50**. \n

                Also check me out on Social Media
                - [YouTube](https://www.youtube.com/@edlabadkarsameer/videos)
                - [LinkedIn](https://www.linkedin.com/in/sameer-edlabadkar-43b48b1a7/)
                - [GitHub](https://github.com/edlabadkarsameer)
              If you are facing any issue while working feel free to mail me on **edlabadkarsameer@gmail.com**

                ''')
    
elif app_mode == 'Sign Language to Text':
   
    st.title('Sign Language to Text')
    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")
    if record:
        st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')
    sameer=""
    st.markdown(' ## Output')
    st.markdown(sameer)

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO

    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.sidebar.markdown('---')
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
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4




    
    detector = HandDetector(maxHands=1)
    classifier = Classifier("keras_model.h5", "labels.txt")

    offset = 20
    imgSize = 300

    
    counter = 0

    labels = ["A", "B", "C"]

    while True:
        success, img = vid.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                        (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset),
                        (x + w+offset, y + h+offset), (255, 0, 255), 4)


            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        cv2.imshow("Image", imgOutput)
        cv2.waitKey(1)
    
else:
    st.title('Speech to Sign Language (The System use Indian Sign Language)')
    # initialize the speech recognition engine
    # initialize the speech recognition engine
    r = sr.Recognizer()


    # define function to display sign language images
    def display_images(text):
        # get the file path of the images directory
        img_dir = "images/"

        # initialize variable to track image position
        image_pos = st.empty()

        # iterate through the text and display sign language images
        for char in text:
            if char.isalpha():
                # display sign language image for the alphabet
                img_path = os.path.join(img_dir, f"{char}.png")
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

