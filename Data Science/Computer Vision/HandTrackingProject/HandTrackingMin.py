import cv2
import mediapipe as mp
import time

cap =cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime =0
cTime =0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks :
        for handlms in results.multi_hand_landmarks:
            for id,lm in enumerate(handlms.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                print(id,cx,cy)
                #if id ==4:
                #cv2.circle(img,(cx,cy),12,(255,0,255),cv2.FILLED)
            mpDraw.draw_landmarks(img,handlms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fbs= 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fbs)),(10,70), cv2.FONT_HERSHEY_COMPLEX ,1.3 , (255,0,255),2)
    cv2.imshow("image", img)

    if cv2.waitKey(30) & 0xff==ord('x'):
        break

cap.release()
cv2.destroyWindow()