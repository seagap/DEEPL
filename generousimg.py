# coding:utf-8
import cv2
import numpy as np
import random as rd
import shelve

imginfo = shelve.open("imginfo")
for i in range(1, 5001):
    x, y = rd.randint(300, 500), rd.randint(300, 500)
    b, d = rd.randint(5, x), rd.randint(5, y)
    a, c = rd.randint(0, b-5), rd.randint(0, d-5)
    print(x, y, a, b, c, d)
    img = np.ones((x, y, 3), dtype=np.uint8) * 255
    img[a:b, c:d] = np.zeros((b - a, d - c, 3), np.uint8)
    # cv2.waitKey(0)
    cv2.imwrite('../generousimg/' + str(i) + '.jpg', img)
    templist = {"x": x, "y": y, "a": a, "b": b, "c": c, "d": d,  "cenx": ((c + d) / (2 * y)),
                "ceny": ((a + b) / (2 * x))}
    imginfo[str(i)] = templist
    print(i,templist)
imginfo.close()
