from ultralytics import YOLO
import cv2 
import math
import time
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

classNames = ['Paper', 'Rock', 'Scissors']

model = YOLO(f'runs\\detect\\trainGoogleColab3\\weights\\best.pt')

testType = input("What gesture are you testing? ")
confidenceList = []
testTime = time.time() + 60

while time.time() < testTime:
    success, img = cap.read()
    results = model(img,conf=0.5)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # class name
            cls = int(box.cls[0])
            # put box in cam
            if(classNames[cls] == testType):
                # confidence
                confidence = math.ceil((box.conf[0]*100))/100

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 255, 255)
                thickness = 2
                confidenceStr = str(confidence)

                cv2.putText(img, classNames[cls]+confidenceStr, org, font, fontScale, color, thickness)
                if confidence > 0.5:
                    confidenceList.append(confidence)

            

    
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

sumConfidence = 0
for i in confidenceList:
    sumConfidence = i + sumConfidence

length = len(confidenceList)
averageConfidence = sumConfidence / length
print(averageConfidence)