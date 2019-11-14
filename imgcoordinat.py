# coding: utf-8
import cv2
import shelve
import numpy as np

img = cv2.imread("s.png")
coordnt = np.zeros(shape=(2, 3))
counter = 0
hights = img.shape[0]
widths = img.shape[1]

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(0, 3):
            if coordnt[0][i] == 0:
                coordnt[0][i], coordnt[1][i] = x, y
                counter = i
                texts = "%d,%d" % (coordnt[0][i], coordnt[1][i])
                cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
                if i == 1:
                    cv2.putText(img, texts, (x, y), cv2.FONT_HERSHEY_PLAIN,
                                1.0, (0, 0, 0), thickness=1)
                    cv2.rectangle(img, (int(coordnt[0][0]), int(coordnt[1][0])), (int(coordnt[0][1]), int(coordnt[1][1])), (0, 0, 255), 2)
                else:
                    cv2.putText(img, texts, (x, y), cv2.FONT_HERSHEY_PLAIN,
                                1.0, (0, 0, 0), thickness=1)
                break
        cv2.imshow("image", img)
        print(coordnt)


cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)

# while (True):
#     try:
#         cv2.waitKey(100)
#     except Exception:
#         cv2.destroyAllWindows()
#         break

s = cv2.waitKey(0)
cv2.destroyAllWindows()
