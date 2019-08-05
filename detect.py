from keras.models import load_model
import cv2 as cv
import numpy as np

model = load_model("D:\cv\DEEPL\s.h5")
src0 = cv.imread('0.jpg')
src0 = src0 / 255.
src1 = cv.imread('1.jpg')
src1 = src1 / 255.
src2 = cv.imread('2.jpg')
src2 = src2 / 255.
src2 = cv.resize(src2, 20, 20)
print(src2)
test = np.array([src0, src1, src2])
# test = np.append(test, [[src2]])
print(test)
# print("result:")
# result = model.predict(test)
# print(result)
# print(np.argmax(model.predict(test)))
