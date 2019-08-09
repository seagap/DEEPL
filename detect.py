from keras.models import load_model
import cv2 as cv
import numpy as np

# model = load_model("D:\cv\DEEPL\s.h5")#正负样本
model = load_model("D:\cv\DEEPL\s5.h5")  # 单正样本
print(model.summary())

src0 = cv.imread('0.jpg')
src0 = cv.resize(src0, (20, 20))
src0 = src0 / 255.
test1 = np.array([src0])
src1 = cv.imread('3.jpg')
src1 = cv.resize(src1, (20, 20))
src1 = src1 / 255.
test2 = np.array([src1])
src2 = cv.imread('7.jpg')
src2 = cv.resize(src2, (20, 20))
src2 = src2 / 255.
test3 = np.array([src2])
test = np.array([src0])
print(test)
print("result:")
# result = model.predict(test)
# print(result)
print(model.predict(test, batch_size=1, verbose=1))
# print(np.argmax(model.predict(test2)))
# print(np.argmax(model.predict(test3)))
