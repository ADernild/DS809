import keras
import numpy as np
import tensorflow as tf
import sklearn
import os
import glob
import matplotlib
import numpy
from numpy import shape
from matplotlib import image
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import ImageDataGenerator
from numpy import asarray
import datetime
from keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from glob import glob
from shutil import copy, move
from PIL import Image
import glob
import pandas as pd
from keras.models import sequential

imagex = 250
imagey = 250
batch = 30

model = keras.Sequential()
        #Layers, Pooling, Dropout and Density
layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(imagex, imagey, 3)),
layers.Conv2D(32, 3, padding='same', activation='relu'),
layers.MaxPooling2D(2,2),
layers.Dropout(0.2),

layers.Conv2D(64, 3, padding='same', activation='relu'),
layers.Conv2D(64, 3, padding='same', activation='relu'),
layers.MaxPooling2D(2, 2),
layers.Dropout(0.2),

layers.Conv2D(128, 3, padding='same', activation='relu'),
layers.Conv2D(128, 3, padding='same', activation='relu'),
layers.MaxPooling2D(2, 2),
layers.Dropout(0.2),

layers.Dense(1, activation='sigmoid')
model.fit(imagex, imagey, batch_size=30, epochs=100)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

#test