import cv2
import time
import os
import HandTrackingModule as htm 
wCam = 720
hCam = 500
cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime =0
folderPath ='fingers_img'
myList= os.listdir(folderPath)
print(myList)

overLaylist= []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    #print(f'{folderPath}/{imPath}')
    overLaylist.append(image)
print(len(overLaylist))

detector =  htm.HandDetector(detectionCon=0.75)
tipIds =[ 4, 8, 12, 16, 20]
while True:
    success , img = cap.read()
    img = detector.findHands(img)
    limlist = detector.findPosition(img , draw=False)
    #print(limlist)

    if len(limlist) != 0:
        fingers =[]

        # Thumb
        if limlist[tipIds[0]][1] > limlist[tipIds[0]-1][2]:  # Check distance to wrist
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers 
        for id in range(1,5):
            if limlist[tipIds[id]][2]<limlist[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        #print(fingers)
        totalFingers =fingers.count(1)
        print(totalFingers)

        h, w, c =overLaylist[totalFingers-1].shape
        img[0:h,0:w] = overLaylist[totalFingers-1]
        cv2.rectangle(img,(20,255),(170,425),(0,255,0),cv2.FILLED)
        cv2.putText(img,str(totalFingers),(45,375),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),25)
    cTime = time.time()
    fps= 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,f'FPS: {int(fps)}',(400,70), cv2.FONT_HERSHEY_PLAIN ,2.5, (255,0,0),3)

    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xff==ord('x'):
        break