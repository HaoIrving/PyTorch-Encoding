
import cv2
import numpy as np
 
# 1.读取图片
img = cv2.imread(r"D:\...\1.png")
cv2.imshow('img', img)
 
# 2. cv2.MORPH_OPEN 先进行腐蚀操作，再进行膨胀操作
kernel = np.ones((18, 18), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imshow('opening', opening)
 
# 3. cv2.MORPH_CLOSE 先进行膨胀，再进行腐蚀操作
kernel = np.ones((18, 18), np.uint8)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

cv2.imshow('closing', closing)

