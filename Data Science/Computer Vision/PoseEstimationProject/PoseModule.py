import cv2
import mediapipe as mp
import time
import math as m
class poseDetector():
    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

    def findPose(self , img, draw =True):

        imRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imRGB)
        #print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img
    
    def findPosition(self, img, draw =True):
        self.limlist = []
        if self.results.pose_landmarks:
            for id , lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c=img.shape
                #print(id ,lm)
                cx , cy = int(lm.x*w), int(lm.y*h)
                self.limlist.append([id,cx,cy])
                if draw:
                    cv2.circle(img ,(cx,cy), 5 ,(255,0,0), cv2.FILLED)
        return self.limlist
    def findAngle(self, img , p1,p2 ,p3,draw =True):
        # Get the landmarks
        x1 ,y1 = self.limlist[p1][1:]
        x2 ,y2 = self.limlist[p2][1:]
        x3 ,y3 = self.limlist[p3][1:]

        # Calculate the angle
        angle = m.degrees(m.atan2(y2-y3,x2-x3)-m.atan2(y2-y1,x2-x1))
        if angle <0 :
            angle += 360

        # draw
        if draw:
            cv2.line(img , (x1,y1), (x2,y2),(0 , 255, 0), 3)
            cv2.line(img , (x3,y3), (x2,y2),(0 , 255, 0), 3)
            cv2.circle(img ,(x1,y1), 10 ,(0,0,255), cv2.FILLED)
            cv2.circle(img ,(x1,y1), 15 ,(0,0,255), 2)
            cv2.circle(img ,(x2,y2), 10 ,(0,0,255), cv2.FILLED)
            cv2.circle(img ,(x2,y2), 15 ,(0,0,255), 2)
            cv2.circle(img ,(x3,y3), 10 ,(0,0,255), cv2.FILLED)
            cv2.circle(img ,(x3,y3), 15 ,(0,0,255), 2)
            cv2.putText(img , str(int(angle)),(x2-50, y2+50), 2,cv2.FONT_HERSHEY_PLAIN ,(0,0,255),2 )

        return angle


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()
    while True:
        success , img = cap.read()
        img = detector.findPose(img)
        limlist = detector.findPosition(img)
        if len(limlist) !=0:
            print(limlist)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)),(70,50),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,0),3)
        cv2.imshow("image",img)
        if cv2.waitKey(10) & 0xff ==ord("x"):
            break


if __name__ =="__main__":
    main()