import cv2

face_Cascade = cv2.CascadeClassifier ('haarcascade_frontalface_default.xml')

#image = cv2.imread('smile.jpeg')
#image = cv2.imread('faces2.jpg')
#image = cv2.imread('faces3.png')
image = cv2.imread('faces4.jpg')

recolor = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

faces = face_Cascade.detectMultiScale(recolor,1.1,4) #scale factor, minimum number of neighboring rectangles required for detection

for (x,y,width,height) in faces:
    cv2.rectangle(image,(x,y),(x+width,y+height),(0,0,255),2)

cv2.imshow('image',image)
cv2.waitKey()
