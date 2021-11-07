#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Libraries
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import datetime


# Set path to this file location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#%% ImageDataGenerator

image_size = [200, 200]
batch = 32

# train_datagen = ImageDataGenerator(rotation_range=40,
#                                    rescale=1./255,
#                                    shear_range=0.1,
#                                    zoom_range=0.2,
#                                    horizontal_flip= True,
#                                    vertical_flip=True,
#                                    width_shift_range=0.1,
#                                    height_shift_range=0.1,
#                                    brightness_range=(0.3, 1.5))

# train_datagen = ImageDataGenerator(rescale=1./255,
#                                     horizontal_flip=True) # Best one setting, however, might overfit at high epochs > 25

train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=30,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    brightness_range=[0.5, 1.3]) # Best combination of settings, so far.


test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    'train',
    target_size=image_size,
    batch_size=batch,
    class_mode='binary',
    seed=1338)

val_gen = test_datagen.flow_from_directory(
        'val',
        target_size=image_size,
        batch_size=batch,
        class_mode='binary',
        seed=1338)

test_gen = test_datagen.flow_from_directory(
    'test',
    target_size=image_size,
    batch_size=batch,
    class_mode='binary',
    seed=1338)

full_gen = train_datagen.flow_from_directory(
    'full_dataset',
    target_size=image_size,
    batch_size=batch,
    class_mode='binary',
    seed=1338)

#%% Model Initialization

model = keras.Sequential([
    # First convolution
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(image_size + [3])),
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
    layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)),
    layers.Dropout(.5),
    
    # Binary output layer
    layers.Dense(1, activation='sigmoid')
    ])

model.summary() # model summary

opt = keras.optimizers.Adam(learning_rate=0.0001) #trying a smaller learning rate

model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy']) # compiling model

#%% Training model

# Callbacks for tensorboard 
tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs") # tensorboard --logdir ./logs

# Step sizes for train, validation and testing
STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VAL=val_gen.n//val_gen.batch_size
STEP_SIZE_TEST=test_gen.n//test_gen.batch_size

epochs = 10

# Fitting model
hist = model.fit(
    train_gen,
    steps_per_epoch=STEP_SIZE_TRAIN,
    epochs=epochs,
    validation_data=val_gen,
    validation_steps=STEP_SIZE_VAL,
    callbacks=[tensorboard_callback])

#%% Number of images generated with current model setup
print(f'Number of images generated: {epochs*batch*STEP_SIZE_TRAIN}')

#%% Model evaluation
model.evaluate(test_gen, steps=STEP_SIZE_TEST) # accuracy 0.8086

#%% Training on train and val together

# Step sizes for train, validation and testing
STEP_SIZE_FULL=full_gen.n//full_gen.batch_size
STEP_SIZE_TEST=test_gen.n//test_gen.batch_size

epochs = 10

# Fitting model
hist = model.fit(
    full_gen,
    steps_per_epoch=STEP_SIZE_FULL,
    epochs=epochs)

#%% Number of images generated with current model setup
print(f'Number of images generated: {epochs*batch*STEP_SIZE_FULL}')

#%% Model evaluation
model.evaluate(test_gen, steps=STEP_SIZE_TEST) # accuracy 0.8398

#%% Saving Model
model.save('aid/final_model.h5')

import matplotlib as plt
from matplotlib import pyplot
import sys


#History is changed to hist based on the .py file model_2
#If model_2 is used as is, this .py file should have the proper imports from model_2

loss_train = hist.history['train_loss']
loss_val = hist.history['val_loss']
epochs = range(1,150)

"""plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show() #or is it plt.print() ???

loss_train = hist.hist['accuracy']
loss_val = hist.hist['val_accuracy']
epochs = range(1,150)"""

plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#The plts above will produce standard line plots.


import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge
import model
from matplotlib import pyplot as plt

def learning_summary(hist):
    #Plotting loss
    plt.subplot(210)
    plt.title('Cross entropy loss')
    plt.plot(hist.hist['loss'], color = 'red', label = 'training')
    plt.plot(hist.hist['validation_loss'], color = 'blue', label = 'test')
    #Plotting the accuracy
    plt.subplot(211)
    plt.title('Accuracy of classification')
    plt.plot(hist.hist['accuracy'], color = 'orange', label = 'training')
    plt.plot(hist.hist['validation_accuracy'], color = 'green', label = 'test')
    #Plot saved to file
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + 'plt.png')
    plt.close()

    #If colorblindness is an issue, the colors chosen for the graphs needs change.