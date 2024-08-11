import  cv2
import  mediapipe as mp
import time

cap= cv2.VideoCapture(0)
pTime =0

mpDraw =mp.solutions.drawing_utils
mpFace = mp.solutions.face_detection
face = mpFace.FaceDetection(0.75)

while True:
    success , img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face.process(imgRGB)
    print(results)

    if results.detections:
        for id , detection in enumerate(results.detections):
            print(id, detection)
            print(detection.location_data.relative_bounding_box)
            ih, iw, ic =img.shape
            bboxc = detection.location_data.relative_bounding_box
            bbox = int(bboxc.xmin*iw),int(bboxc.ymin*ih),int(bboxc.width*iw), int(bboxc.height*ih)
            cv2.rectangle(img ,bbox, (255,0,255),3)
            cv2.putText(img , f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,2, (255,0,255),2)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img , f'FPS: {int(fps)}',(30,60),cv2.FONT_HERSHEY_PLAIN,0.9, (100,250,0),2)
    cv2.imshow("image",img)
    if cv2.waitKey(10) & 0xff ==ord('x'):
        break