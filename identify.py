from keras.utils import to_categorical
from keras.utils import get_file
import numpy as np
from keras import models
from keras import layers
path = get_file('mnist', origin='https://s3.amazonaws.com/img-datasets/mnist.npz',cache_dir='D:/')
f = np.load(path)
train_images, train_labels = f['x_train'], f['y_train']
test_images, test_labels = f['x_test'], f['y_test']

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',
loss='sparse_categorical_crossentropy',
metrics=['accuracy']
                )
network.fit(train_images, train_labels, epochs=5, batch_size=128)