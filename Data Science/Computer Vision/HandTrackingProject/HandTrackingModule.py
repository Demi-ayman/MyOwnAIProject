import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self,mode =False ,maxHand =2,detectionCon = 0.5,trackCon =0.5):
        self.mode = mode
        self.maxHand = maxHand
        #self.modelComplexity = modelComplexity
        self.detectionCon =detectionCon
        self.trackCon =trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHand,int(self.detectionCon),int(self.trackCon))
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds =[ 4, 8, 12, 16, 20]

    def findHands(self,img , draw =True):
        imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks :
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handlms, self.mpHands.HAND_CONNECTIONS)
        return img
    def findPosition(self, img , handNo = 0,draw =True):
        self.limlist = []
        if self.results.multi_hand_landmarks:
            myHand= self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                #print(id,cx,cy)
                self.limlist.append([id, cx,cy])
                #if id ==4:
                if draw:
                    cv2.circle(img,(cx,cy),12,(255,0,255),cv2.FILLED)
        return self.limlist
    
    def fingersUP(self):
        fingers =[]

        # Thumb
        if self.limlist[self.tipIds[0]][1] > self.limlist[self.tipIds[0]-1][2]:  # Check distance to wrist
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers 
        for id in range(1,5):
            if self.limlist[self.tipIds[id]][2]<self.limlist[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


def main():
    pTime =0
    cTime =0
    cap =cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        limset = detector.findPosition(img)
        if len(limset) !=0:
            print(limset[4])
        cTime = time.time()
        fbs= 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fbs)),(10,70), cv2.FONT_HERSHEY_COMPLEX ,1.3 , (255,0,255),2)
        cv2.imshow("image", img)

        if cv2.waitKey(10) & 0xff==ord('x'):
            break


if __name__ == "__main__":
    main()