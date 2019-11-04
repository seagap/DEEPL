# -*- coding: UTF-8 -*-
import cv2 as cv
import tkinter

def call(event):
    video.release()  # 关闭相机
video = cv.VideoCapture("test.mp4")
while (True):
    rets, frame = video.read()  # 捕获一帧图像
    if rets:
        test=cv.imshow('DVR', frame)
        cv.waitKey(25)
    else:
        break

test.bind("<Key>", call)
test.focus_set()


#
# print("Press 'D' to exit...")
#
# while True:
#     if ord(msvcrt.getch()) in [68, 100]:
#         break
# cv.destroyAllWindows()
#
#
# def call(event):
#     print(event.keysym)  # 打印按下的键值
# win = tkinter.Tk()
# frame = tkinter.Frame(win, width=200, height=200)
# frame.bind("<Key>", call)  # 触发的函数
# frame.focus_set()  # 必须获取焦点
# frame.pack()
# win.mainloop()