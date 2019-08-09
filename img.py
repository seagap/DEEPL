from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
import matplotlib.pyplot as plt
from keras import optimizers
import os

# os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息  
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error   
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # 只显示 Erro

train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_dir = 'D:\cv\\trainset'
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(20, 20),
    batch_size=1000,
    class_mode='binary')

validation_dir = 'D:\cv\\validation'
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(20, 20),
    batch_size=1000,
    class_mode='binary')

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

for data_batch, labels_batch in validation_generator:
    print('vadata batch shape:', data_batch.shape)
    print('valabels batch shape:', labels_batch.shape)
    break

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu',
                        input_shape=(20, 20, 3)))
model.add(layers.Flatten())
model.add(layers.Dense((1,1), activation='sigmoid'))
print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
print(train_generator.class_indices)
history = model.fit_generator(
    generator=train_generator,
    verbose=1,
    steps_per_epoch=1000,
    epochs=1,
    validation_data=validation_generator,
    validation_steps=1000)
print(history)
model.save('cats_and_dogs_small_1.h5')

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()
# test
# ok
