# -*- coding: UTF-8 -*-
import cv2 as cv
import random as rd
import tkinter


def call(event):
    video.release()  # 关闭相机


video = cv.VideoCapture("test.mp4")
while (True):
    rets, frame = video.read()  # 捕获一帧图像
    cv.rectangle(frame, ( rd.randint(1,800), rd.randint(1,400)), (rd.randint(1,800), rd.randint(1,400)), (rd.randint(1,255), rd.randint(1,255), rd.randint(1,255)), thickness=3, lineType=8, shift=0)
    if rets:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        circle1 = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=100,
                                  maxRadius=200)  # 把半径范围缩小点，检测内圆，瞳孔
        print(circle1)
        test = cv.imshow('DVR', frame)
        flag = cv.waitKey(0)
        if (flag == 32):
            continue
        if (flag == 27):
            break
    else:
        break
frame = video.read()  # 捕获一帧图像
print(frame[1].shape)
cv.waitKey(0)
cv.destroyAllWindows()
