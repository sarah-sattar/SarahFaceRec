import cv2
import numpy as np
from matplotlib import pyplot as plt

img_ref = cv2.imread('frank1.jpg')
img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20, 10))
plt.imshow(img_ref)
plt.title('Reference Image')
plt.show()

gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
keypoint_ref, descriptors_ref = sift.detectAndCompute(gray_ref, None)

images = ['sterling2.jpg', 'sterling3.jpg', 'sterling4.jpg', 'chrisrock2.jpg','chrisrock3.jpg',
'chrisrock4.jpg', 'idris2.jpg','idris3.jpg','idris4.jpg','frank2.jpg','frank3.jpg','frank4.jpg','kevin2.jpg','kevinhart3.jpg','kevinhart4.jpg']

for file in images:
    img_test = cv2.imread(file)
    img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

    gray_test = cv2.cvtColor(img_test, cv2.COLOR_RGB2GRAY)

    keypoint_test, descriptors_test = sift.detectAndCompute(gray_test, None)

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(descriptors_ref, descriptors_test, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    print(len(good_matches))

    img_matches = cv2.drawMatches(
        gray_ref, keypoint_ref, gray_test, keypoint_test, good_matches, None, flags=2
    )


    print(f"{file} has {len(good_matches)} matches")


    plt.figure(figsize=(40, 20))
    plt.imshow(img_matches)
    plt.show()
