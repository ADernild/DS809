#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 15:37:23 2021

@author: adernild
"""

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
