import  cv2
import  mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectorCon=0.5):
        self.minDetectorCon=minDetectorCon
        self.mpDraw =mp.solutions.drawing_utils
        self.mpFace = mp.solutions.face_detection
        self.face = self.mpFace.FaceDetection(minDetectorCon)
    
    def findFaces(self,img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face.process(imgRGB)
        #print(self.results)
        bboxs = []
        if self.results.detections:
            for id , detection in enumerate(self.results.detections):
                ih, iw, ic =img.shape
                bboxc = detection.location_data.relative_bounding_box
                bbox = int(bboxc.xmin*iw),int(bboxc.ymin*ih),int(bboxc.width*iw), int(bboxc.height*ih)
                bboxs.append([id,bbox,detection.score])
                if draw:
                    img =self.fancyDraw(img, bbox)
                    cv2.putText(img , f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,2, (255,0,255),2)
        return img , bboxs
    def fancyDraw(self, img, bbox, l=30,t=5, rt=1):
        x,y,w,h =bbox
        x1,y1 = x+w, y+h
        cv2.rectangle(img ,bbox, (255,0,255),rt)
        # Top left x,y
        cv2.line(img , (x,y),(x+l,y),(255,0,255),t)
        cv2.line(img , (x,y),(x,y+l),(255,0,255),t)
        # Top right x1,y1
        cv2.line(img , (x1,y),(x1-l,y),(255,0,255),t)
        cv2.line(img , (x1,y),(x1,y+l),(255,0,255),t)
        # Bottom left x,y
        cv2.line(img , (x,y1),(x+l,y1),(255,0,255),t)
        cv2.line(img , (x,y1),(x,y1-l),(255,0,255),t)
        # Bottom right x1,y1
        cv2.line(img , (x1,y1),(x1-l,y1),(255,0,255),t)
        cv2.line(img , (x1,y1),(x1,y1-l),(255,0,255),t)

        return img
        
    

def main():
    cap= cv2.VideoCapture(0)
    pTime =0
    detector = FaceDetector()
    while True:
        success , img = cap.read()
        img,bboxs = detector.findFaces(img)
        print(bboxs)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img , f'FPS: {int(fps)}',(30,60),cv2.FONT_HERSHEY_PLAIN,0.9, (100,250,0),2)
        cv2.imshow("image",img)
        if cv2.waitKey(10) & 0xff ==ord('x'):
            break


if __name__ == "__main__":
    main()