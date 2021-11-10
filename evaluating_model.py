#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 15:37:23 2021

@author: adernild
"""

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from numpy import expand_dims

#%%
model_1 = keras.models.load_model('aid/final_model.h5')

model_2 = keras.models.load_model('aid/final_model_2.h5')

#%%
image_size = [200, 200]
batch = 32

test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    'test',
    target_size=image_size,
    batch_size=batch,
    class_mode='binary',
    seed=1338)

#%% 
STEP_SIZE_TEST=test_gen.n//test_gen.batch_size

model_1.evaluate(test_gen, steps=STEP_SIZE_TEST) # accuracy 0.8398
model_2.evaluate(test_gen, steps=STEP_SIZE_TEST) # accuracy 0.8125

#%%
import numpy as np
from keras.preprocessing import image

einstein = 'bonus/dog/dog.1773.jpg' # einstein
cat_dog = 'bonus/dog/dog.7.jpg' # cat and dog
cat_dog_2 = 'bonus/cat/cat.724.jpg' # cat and dog
toy_cat = 'bonus/cat/cat.92.jpg'

def image_prep(path):
    img_width, img_height = 200, 200
    img = image.load_img(path, target_size = (img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    return img

if model_1.predict(image_prep(einstein)) < 0.51:
    print("Einstein is a cat!")
else:
    print("Einstein is a dog!")

if model_1.predict(image_prep(cat_dog)) < 0.51:
    print("It's a cat!")
else:
    print("It's a dog!")
    
if model_1.predict(image_prep(cat_dog_2)) < 0.51:
    print("It's a cat!")
else:
    print("It's a dog!")
    
if model_1.predict(image_prep(toy_cat)) < 0.51:
    print("It's a cat!")
else:
    print("It's a dog!")
