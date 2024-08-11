import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm
pTime =0 # previous Time 
cTime =0 # current Time
cap =cv2.VideoCapture(0)
detector = htm.HandDetector()
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    limset = detector.findPosition(img )
    if len(limset) !=0:
        print(limset[4])
    cTime = time.time()
    fbs= 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fbs)),(10,70), cv2.FONT_HERSHEY_COMPLEX ,1 , (255,0,255),2)
    cv2.imshow("image", img)

    if cv2.waitKey(10) & 0xff==ord('x'):
        break