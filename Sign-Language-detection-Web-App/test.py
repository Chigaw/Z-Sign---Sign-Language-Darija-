def DarijaSign():
        st.title('Darija Morroco Sign Langage')
        np.set_printoptions(suppress=True)
        model = load_model("keras_model1.h5", compile=False)
        class_names = open("labels.txt", "r").readlines()

        # Capture vidéo à partir de la caméra
        cap = cv2.VideoCapture(0)

        # Configuration des paramètres de la caméra
        cap.set(3, 320)  # Set width to 320
        cap.set(4, 240)    # Hauteur de la caméra 240

        st.subheader('Click the button below to activate the camera and start facial recognition.')
        btn_start_camera = st.button("Activer la caméra")
      

        if btn_start_camera:
            st.warning("La caméra est activée. Pour arrêter, cliquez sur le bouton 'Arrêter la caméra'.")
            # Placeholder to display the webcam feed
            placeholder = st.empty()
            predictc = ""
            accumulated_predictions = []
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

                # Display the webcam feed
                # placeholder.image(frame, channels="BGR", use_column_width=True, output_format="BGR")
                placeholder.image(frame, channels="BGR", use_column_width=True, output_format="BGR")
                # Print prediction and confidence score
                # st.write(class_name[2:], end="")
                # print(class_name[2:], end="")
                # print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")


                # if (predictc != class_name[2:]) and (confidence_score > 0.8):
                #     predictc = class_name[2:]
                #     st.subheader(predictc + '\t')
                #     print("Class:", class_name[2:], end="")
                #     print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
                
              
  
            
    # ... (your existing code)

                if (predictc != class_name[2:]) and (confidence_score > 0.8):
                    predictc = class_name[2:]
                    
                    accumulated_predictions.append(predictc)
                    st.subheader('\t'.join(accumulated_predictions))
                
                # Display the accumulated predictions
                
                            
                # Sleep for a short time to control the frame rate
                time.sleep(0.1)