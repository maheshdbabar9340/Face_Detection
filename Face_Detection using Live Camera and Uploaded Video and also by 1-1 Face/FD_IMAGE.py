#Face Detection for Images

#import Packages
import cv2
import face_recognition
 
#Enter Path of Image1  
img = face_recognition.load_image_file('C:\\Users\\mahes\\test\\face\\ImageBasic\\2.jpeg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#Enter Path of Image2
imgTest = face_recognition.load_image_file('C:\\Users\\mahes\\test\\face\\ImageBasic\\mahesh.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

#Face Detection on Image2 using Image1
faceLoc = face_recognition.face_locations(img)[0]
encodeImg = face_recognition.face_encodings(img)[0]
cv2.rectangle(img,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
 
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
 
results = face_recognition.compare_faces([encodeImg],encodeTest)
faceDis = face_recognition.face_distance([encodeImg],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
 
cv2.imshow('Image',img)
cv2.imshow('Test Result',imgTest)
cv2.waitKey(0)