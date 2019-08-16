# 打开图片（图片放在工程目录 img 文件夹下）
import keras as kr
import PIL as plt
import numpy as np
img = plt.Image.open('img/test.jpg')

# 转换成灰度图是 "L"，转换成RGB是"RGB"
img_gray = img.convert("L")

# 因为 keras 要求 2D 卷积层的输入的形状是 (宽, 高, 通道数)，这里只有灰度值一个通道，所以对维度进行扩展
img_gray_array = np.expand_dims(np.array(img_gray), axis=2)
# img_gray_array.shape = (64, 64, 1)
# 创建模型
model = kr.Sequential()

# 定义卷积核
def jk_kernel(shape):
    return np.expand_dims(np.expand_dims(kernel, axis=2), axis=2)

# 添加卷积层
model.add(kr.layers.Conv2D(filters=1, kernel_size=3, kernel_initializer=jk_kernel, strides=1, padding='same', name='conv2d', input_shape=img_gray_array.shape))

# 编译模型
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
conv2d_layer_model = kr.Model(inputs=model.input,
                                     outputs=model.get_layer('conv2d').output)
conv2d_output = conv2d_layer_model.predict(np.expand_dims(img_gray_array, axis=0))
# conv2d_output.shape = (1, 64, 64, 1)
# 缩减维度
img_gray_array_squeeze = np.squeeze(conv2d_output)
# 保存结果，也可以用 show() 方法来显示图片
plt.Image.fromarray(img_gray_array_squeeze).convert('RGB').save(fp='img/edge.png')