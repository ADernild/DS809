#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Libraries
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#%% ImageDataGenerator

image_size = [200, 200]
batch = 32

train_datagen = ImageDataGenerator(rotation_range=40,
                                   rescale=1./255,
                                   shear_range=0.1,
                                   zoom_range=0.2,
                                   horizontal_flip= False,
                                   vertical_flip=True,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1)

train_gen = train_datagen.flow_from_directory(
    'train',
    target_size=image_size,
    batch_size=batch,
    class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)

val_gen = test_datagen.flow_from_directory(
        'val',
        target_size=image_size,
        batch_size=batch,
        class_mode='binary')

test_gen = test_datagen.flow_from_directory(
    'test',
    target_size=image_size,
    batch_size=batch,
    class_mode='binary')

#%% Model Initialization

model = keras.Sequential([
    # First convolution
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
    layers.MaxPooling2D(2,2), # halving the image size 
    
    # Second convolution
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    # Third convolution
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    # Fourth convolution
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    # Fifth convolution
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    # Flatten results to feed into a Deep Nerual Net
    layers.Flatten(),
    
    # 512 neuron hidden layer
    layers.Dense(512, activation='relu'),
    
    # Binary output layer
    layers.Dense(1, activation='sigmoid')
    ])

model.summary() # model summary

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']) # compiling model

#%% Training model

# Callbacks for tensorboard 
tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs") # tensorboard --logdir ./logs

# Step sizes for train, validation and testing
STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VAL=val_gen.n//val_gen.batch_size
STEP_SIZE_TEST=test_gen.n//test_gen.batch_size

# Fitting model
hist = model.fit(
    train_gen,
    steps_per_epoch=STEP_SIZE_TRAIN,
    epochs=30,
    validation_data=val_gen,
    validation_steps=STEP_SIZE_VAL,
    callbacks=[tensorboard_callback])

#%% Model evaluation
model.evaluate(test_gen, steps=STEP_SIZE_TEST) # accuracy 0.7305

#%% Saving Model
model.save('aid/first_model.h5')