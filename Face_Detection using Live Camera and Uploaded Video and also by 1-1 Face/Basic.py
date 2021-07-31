
import cv2
import numpy as np
import face_recognition
import os


path='ImageBasic'
images=[]
classNames=[]
myList=os.listdir(path)
print(myList)

for cl in myList:
  currImg= cv2.imread(f'{path}/{cl}')
  images.append(currImg)
  classNames.append(os.path.splitext(cl)[0])

print(classNames)

def findEncoding(images):
  encodeList=[]
  for img in images:
    img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    encode = face_recognition.face_encodings(img)[0]
    encodeList.append(encode)
  return encodeList

encodeListKnown = findEncoding(images)
#print(len(encodeListKnown))
print("encoding complete")

cap = cv2.VideoCapture(0)

while True:
    success, img =cap.read()
    imgS = cv2.resize(img, (0,0),None,0.25,0.25)
    imgS= cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    faceCurrFrame = face_recognition.face_locations(imgS)
    encodeCurrFrame = face_recognition.face_encodings(imgS,faceCurrFrame)


    for encodeFace,faceLoc in zip(encodeCurrFrame,faceCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex= np.argmin(faceDis)

        if matches[matchIndex]:
            name =classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1=faceLoc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)

            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)








