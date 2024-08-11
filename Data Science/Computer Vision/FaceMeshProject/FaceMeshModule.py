import cv2
import mediapipe as mp
import time


class FaceMeshDetector():
    def __init__(self,staticMode = False ,maxFace = 2,minDetectionCon = 0.5,minTrackCon =0.5):
        self.staticMode =staticMode
        self.maxFace =maxFace
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces =2)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img,draw = True):
        self.imgRGB  =cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for facelms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, facelms, self.mpFaceMesh.FACEMESH_TESSELATION,self.drawSpec,self.drawSpec)
                face = []
                for id, lm in enumerate(facelms.landmark):
                    #print(lm)
                    ih, iw, ic= img.shape
                    x, y =int(lm.x*iw),int(lm.y*ih)
                    #cv2.putText(img ,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,0.3,(255,255,0),1)
                    #print(id,x,y)
                    face.append([x,y])
                faces.append(face)
        return img , faces



def main():
    cap = cv2.VideoCapture(0)
    pTime= 0
    detector = FaceMeshDetector()
    while True:
        sucess, img = cap.read()
        img , faces= detector.findFaceMesh(img)
        if len(faces) != 0:
            print(faces[0])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img ,f'FPS: {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
        cv2.imshow("koktie",img)
        if cv2.waitKey(10) & 0xff == ord('x'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ =='__main__':
    main()