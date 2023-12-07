import cv2
import time
import os

# hna ktbo langue ya "english" ya "arabic"
language = "english"
# hna ktbo smitkom
name = "nassim"
# hna ktbo ina sign ghadi tjm3o ha choices li kaynin binisba l english : hello, our_goal_is, to_help, deaf_people
sign = "n3awno"
# hna khtaro ch7al mn tswira bghito
number= 400


# ou hadchi hada no need t9isoh (mn ghir ila bghito), bdlo gha fdok les variables li lfo9 and everything should be good

os.makedirs(f'data/{sign}', exist_ok=True)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 20)

for i in range(number):
    ret, frame = cap.read()

    if ret: 
        cv2.imshow('frame', frame)
        cv2.imwrite(f'data/{sign}/{language}_{name}_{i}.jpeg', frame)  # Save the image
        print(f'Captured Image {i}')

    time.sleep(0.05)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()