import cv2
import numpy as np
import face_recognition

img_pop = face_recognition.load_image_file('frank1.jpg')
img_pop = cv2.cvtColor(img_pop, cv2.COLOR_BGR2RGB)
train_encode = face_recognition.face_encodings(img_pop)[0]



images = ['chrisrock2.jpg','chrisrock3.jpg','chrisrock4.jpg','frank2.jpg','frank3.jpg','frank4.jpg','idris3.jpg','idris2.jpg','idris4.jpg','kevin2.jpg','kevinhart3.jpg','kevinhart4.jpg','sterling2.jpg','sterling3.jpg','sterling4.jpg']



for file in images:
    test = face_recognition.load_image_file(file)
    test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
    test_encode = face_recognition.face_encodings(test)[0]

    compare = face_recognition.compare_faces([train_encode], test_encode)

    if compare[0]:
        print(f"{file} contains the same face as the reference image.")
    else:
        print(f"{file} does not contain the same face as the reference image.")

    face_locations = face_recognition.face_locations(img_pop)
    for faceloc in face_locations:
        top, right, bottom, left = faceloc
        cv2.rectangle(img_pop, (left, top), (right, bottom), (255, 0, 255), 1)

    cv2.imshow('Reference Image', img_pop)
    cv2.waitKey(0)
