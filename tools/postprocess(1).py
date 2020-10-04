import cv2
import numpy as np
import os

array_of_img = []
# 1.读取图片
# img = cv2.imread(r"E:\6 项目申请\电子所\中科星图杯比赛\postprocessing\postprocessing\3_feature.png")
# cv2.imshow('img', img)

directory_name='./postprocessing';
for filename in os.listdir(r"./" + directory_name):
    # print(filename) #just for test
    # img is used to store the image data
    img = cv2.imread(directory_name + "/" + filename)
# 处理结构和半径
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))#椭圆5*5
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(8,8))# 正方形 8*8

# 2. cv2.MORPH_OPEN 先进行腐蚀操作，再进行膨胀操作
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imshow('opening', opening)
 
# 3. cv2.MORPH_CLOSE 先进行膨胀，再进行腐蚀操作
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

cv2.imshow('closing', closing)

array_of_img.append(closing)
        #print(img)
print(array_of_img)



