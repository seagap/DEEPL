from keras.utils import to_categorical
from keras.utils import get_file
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt

path = get_file('mnist', origin='https://s3.amazonaws.com/img-datasets/mnist.npz', cache_dir='D:/')
f = np.load(path)
train_images, train_labels = f['x_train'], f['y_train']
test_images, test_labels = f['x_test'], f['y_test']

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255  # 归一化
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

print(train_images.shape)
print(train_labels)
print(test_images.shape)
print(test_labels)

network = models.Sequential()
network.add(layers.Dense(256,activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
history=network.compile(optimizer='rmsprop',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

history=network.fit(train_images, train_labels, epochs=9, batch_size=128, validation_data=(test_images, test_labels))

history_dict = history.history
print(history_dict.keys())
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, 10)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# network.fit(train_images, train_labels, epochs=5, batch_size=128)

# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)


# test_loss, test_acc = network.evaluate(test_images, test_labels)
# print('test_acc:', test_acc)
# history = network.fit(partial_x_train,
# partial_y_train,
# epochs=20,
# batch_size=512,
# validation_data=(x_val, y_val))

# digit = train_images[4]
# import matplotlib.pyplot as plt
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()

# from keras.utils import to_categorical
# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)
# network.fit(train_images, train_labels, epochs=5, batch_size=128)
