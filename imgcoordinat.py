# coding: utf-8
import cv2
import shelve
import numpy as np

imgid = 1
img = cv2.imread("./img/s (" + str(imgid) + ").jpg")
coordnt = np.zeros(shape=(2, 3))
counter = 0
hights = img.shape[0]
widths = img.shape[1]
traindata = shelve.open("traindata")
print(img.shape[0])

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global counter, imgid, img, coordnt
    if event == cv2.EVENT_LBUTTONDOWN:
        d = x / img.shape[1]
        e = y / img.shape[0]
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        if coordnt[0][2] != 0:
            traindata[str(imgid)] = {"x1": coordnt[0][0], "y1": coordnt[1][0], "x2": coordnt[0][1],"y2": coordnt[1][1],
                                     "x3": coordnt[0][2], "y3": coordnt[1][2], }
            print(imgid,traindata[str(imgid)])
            coordnt = np.zeros(shape=(2, 3))
            imgid += 1
            img = cv2.imread("./img/s (" + str(imgid) + ").jpg")

        else:
            for i in range(0, 2):
                if coordnt[0][i] == 0:
                    coordnt[0][i], coordnt[1][i] = x, y
                    counter = i
                    texts = "%d,%d" % (coordnt[0][i], coordnt[1][i])
                    if i == 1:
                        cv2.putText(img, texts, (x, y), cv2.FONT_HERSHEY_PLAIN,
                                    1.0, (0, 0, 0), thickness=1)
                        cv2.rectangle(img, (int(coordnt[0][0]), int(coordnt[1][0])),
                                      (int(coordnt[0][1]), int(coordnt[1][1])), (0, 0, 255), 2)
                        coordnt[0][2], coordnt[1][2] = (coordnt[0][0] + coordnt[0][1]) / (2 * widths), (
                                    coordnt[1][0] + coordnt[1][1]) / (2 * hights)
                    else:
                        cv2.putText(img, texts, (x, y), cv2.FONT_HERSHEY_PLAIN,
                                    1.0, (0, 0, 0), thickness=1)
                    break
        try:
            cv2.imshow("image", img)

        except Exception:
            print("there is no more img")
            cv2.destroyAllWindows()
cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)
quites = cv2.waitKey(0)
if quites == 27:
    traindata.close()
    cv2.destroyAllWindows()
if quites == 32:
    for i in traindata:
        print(i, traindata[i])
