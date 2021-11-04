#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:39:52 2021

@author: adernild
"""
#%% Libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

#%% ImageDataGenerators

image_size = [300, 300]

train_datagen_1 = ImageDataGenerator(rotation_range=40)

train_datagen_2 = ImageDataGenerator(shear_range=0.1)

train_datagen_3 = ImageDataGenerator(zoom_range=0.3)

train_datagen_4 = ImageDataGenerator(horizontal_flip= True)

train_datagen_5 = ImageDataGenerator(vertical_flip=True)

train_datagen_6 = ImageDataGenerator(width_shift_range=0.1)

train_datagen_7 = ImageDataGenerator(height_shift_range=0.1)

train_datagen_8 = ImageDataGenerator(brightness_range=(0.3,1.5))

train_gen_1 = train_datagen_1.flow_from_directory(
    'train',
    target_size=image_size,
    color_mode='rgb',
    batch_size=1,
    class_mode='binary',
    seed=1337)

train_gen_2 = train_datagen_2.flow_from_directory(
    'train',
    target_size=image_size,
    color_mode='rgb',
    batch_size=1,
    class_mode='binary',
    seed=1337)

train_gen_3 = train_datagen_3.flow_from_directory(
    'train',
    target_size=image_size,
    color_mode='rgb',
    batch_size=1,
    class_mode='binary',
    seed=1337)

train_gen_4 = train_datagen_4.flow_from_directory(
    'train',
    target_size=image_size,
    color_mode='rgb',
    batch_size=1,
    class_mode='binary',
    seed=1337)

train_gen_5 = train_datagen_5.flow_from_directory(
    'train',
    target_size=image_size,
    color_mode='rgb',
    batch_size=1,
    class_mode='binary',
    seed=1337)

train_gen_6 = train_datagen_6.flow_from_directory(
    'train',
    target_size=image_size,
    color_mode='rgb',
    batch_size=1,
    class_mode='binary',
    seed=1337)

train_gen_7 = train_datagen_7.flow_from_directory(
    'train',
    target_size=image_size,
    color_mode='rgb',
    batch_size=1,
    class_mode='binary',
    seed=1337)

train_gen_8 = train_datagen_8.flow_from_directory(
    'train',
    target_size=image_size,
    color_mode='rgb',
    batch_size=1,
    class_mode='binary',
    seed=1337)


generators = [train_gen_1, train_gen_2, train_gen_3, train_gen_4, train_gen_5, train_gen_6, train_gen_7]

#%% Visualizing the different elements of ImageDataGenerator

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,15))
for i in range(4):
     image = next(train_gen_1)[0].astype('uint8')
    
     image = np.squeeze(image)
    
     ax[i].imshow(image)
     ax[i].axis('off')

fig.savefig('plots/rotation_range.png')

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,15))
for i in range(4):
     image = next(train_gen_2)[0].astype('uint8')
    
     image = np.squeeze(image)
    
     ax[i].imshow(image)
     ax[i].axis('off')

fig.savefig('plots/shear_range.png')

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,15))
for i in range(4):
     image = next(train_gen_3)[0].astype('uint8')
    
     image = np.squeeze(image)
    
     ax[i].imshow(image)
     ax[i].axis('off')

fig.savefig('plots/zoom_range.png')

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,15))
for i in range(4):
     image = next(train_gen_4)[0].astype('uint8')
    
     image = np.squeeze(image)
    
     ax[i].imshow(image)
     ax[i].axis('off')
     
fig.savefig('plots/horizontal_flip.png')

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,15))
for i in range(4):
     image = next(train_gen_5)[0].astype('uint8')
    
     image = np.squeeze(image)
    
     ax[i].imshow(image)
     ax[i].axis('off')
     
fig.savefig('plots/vertical_flip.png')

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,15))
for i in range(4):
     image = next(train_gen_6)[0].astype('uint8')
    
     image = np.squeeze(image)
    
     ax[i].imshow(image)
     ax[i].axis('off')
     
fig.savefig('plots/width_shift_range.png')

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,15))
for i in range(4):
     image = next(train_gen_7)[0].astype('uint8')

     image = np.squeeze(image)

     ax[i].imshow(image)
     ax[i].axis('off')

fig.savefig('plots/height_shift_range.png')

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,15))
for i in range(4):
     image = next(train_gen_8)[0].astype('uint8')

     image = np.squeeze(image)

     ax[i].imshow(image)
     ax[i].axis('off')

fig.savefig('plots/brightness_shift_range.png')