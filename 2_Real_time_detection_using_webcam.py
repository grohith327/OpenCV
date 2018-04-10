import cv2
import numpy as np

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():
    rval, img = vc.read()
else:
    rval = False

face_cascade = cv2.CascadeClassifier('C:\\Users\\Rohith\\Documents\\Rohith_Stuff\\haar_cascades\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\Rohith\\Documents\\Rohith_Stuff\\haar_cascades\\haarcascade_eye.xml')
#smile_cascade = cv2.CascadeClassifier('C:\\Users\\Rohith\\Documents\\Rohith_Stuff\\haar_cascades\\haarcascade_smile.xml')
#body_cascade = cv2.CascadeClassifier('C:\\Users\\Rohith\\Documents\\Rohith_Stuff\\haar_cascades\\haarcascade_fullbody.xml')

while rval:
    cv2.imshow("preview", img)
    rval, frame = vc.read()
    
    img = frame
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        '''
        body = body_cascade.detectMultiScale(roi_gray)
        for (bx,by,by,bh) in body:
            cv2.rectangle(roi_color,(bx,by),(bx+bw,by+bh),(0,0,255),2)
        
        smile = smile_cascade.detectMultiScale(roi_gray)
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
        '''
    key = cv2.waitKey(20)
    if key == 27: 
        break
cv2.destroyWindow("preview")
