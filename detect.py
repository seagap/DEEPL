from keras.models import load_model
import cv2 as cv
import numpy as np
import pillow
# model = load_model("D:\cv\DEEPL\s.h5")#正负样本
model = load_model("D:\cv\DEEPL\s10.h5")  # 单正样本 s10 相当于最成功的 ，
# 但是全零居然还不行
#print(model.summary())
sample=np.zeros(shape=(1,20,20,3))
for i in range(1,5):
    strs='.\img\\s (' + str(i) + ').jpg'
    temp = cv.imread(strs)
    temp = cv.resize(temp, (20, 20))
    temp = [temp / 255.]
    sample=np.concatenate((sample,temp),axis = 0)
print(np.shape(sample))
# print(sample)
# print("result:")
# result = model.predict(test)
# print(result)
print(model.predict_classes(sample, batch_size=1, verbose=1))
# print(np.argmax(model.predict(test2)))
# print(np.argmax(model.predict(test3)))