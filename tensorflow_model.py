import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import scipy
import PIL
import cv2
import matplotlib.pyplot as plt
from PIL import Image


IMAGE_HEIGHT = 386
IMAGE_WIDTH = 300
BATCH_SIZE = 10

# Constructor for image generator with some augmentation parameters (rotation)
datagen = ImageDataGenerator(
    rescale=1./255,
    data_format='channels_last',
    validation_split=0.2,
    rotation_range=180,
    dtype=tf.float32
)

# Constructing the dataset for training
train_generator = datagen.flow_from_directory(
    directory=r"images/SB_cropped/",
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='sparse',
    shuffle=True,
    subset='training',
    seed=123456
)
# Constructing the dataset for validation
validation_generator = datagen.flow_from_directory(
    directory=r"images/SB_cropped/",
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='sparse',
    shuffle=True,
    seed=123456,
    subset='validation'
)

# Building the sequential model
model = Sequential()
model.add(Conv2D(input_shape=(386,300,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=32,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=32,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=1024,activation="relu"))
model.add(Dense(units=1024,activation="relu"))
model.add(Dense(units=5,activation="softmax")) # Units reflects the amount of categories that we want at the end

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=15,
    steps_per_epoch=53,
    validation_data=validation_generator,
    validation_steps=13
)


# Plotting the accuracy of the model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.3, 0.7])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(train_generator, verbose=2)
test_loss, test_acc = model.evaluate(validation_generator, verbose=2)
