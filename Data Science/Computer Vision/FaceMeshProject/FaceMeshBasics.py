import cv2
import mediapipe as mp
import time
cap = cv2.VideoCapture(0)
pTime= 0
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces =2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
while True:
    sucess, img = cap.read()
    imgRGB  =cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, facelms, mpFaceMesh.FACEMESH_TESSELATION,drawSpec,drawSpec)
        for id, lm in enumerate(facelms.landmark):
            #print(lm)
            ih, iw, ic= img.shape
            x, y =int(lm.x*iw),int(lm.y*ih)
            print(id,x,y)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img ,f'FPS: {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
    cv2.imshow("koktie",img)
    if cv2.waitKey(10) & 0xff == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()