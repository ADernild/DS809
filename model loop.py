# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:03:34 2021

@author: Max
"""


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.preprocessing import StandardScaler
from numpy.random import seed # to set seed
from tensorflow.random import set_seed # to set seed
import pandas as pd
import os
import tensorflow as tf
#from tensorflow.python.client import device_lib

# Set path to this file location
path = "C:/Users/Max/Documents/python/ds809-project"
os.chdir(path)
#os.path.dirname(os.path.abspath("__file__"))
print(os.getcwd())
#os.chdir('DS809 Deep learning project')
#%%
# Apply seed
seed_value = 1338
set_seed(seed_value)
seed(seed_value)

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

tf.test.is_built_with_cuda()

print(tf.config.list_physical_devices('GPU'))

#%%

# Kopieret fra model_2.py 4/11 ~~11.30
#%% ImageDataGenerator

image_size = [200, 200]
batch = 32

train_datagen = ImageDataGenerator(rotation_range=40,
                                   rescale=1./255,
                                   shear_range=0.1,
                                   zoom_range=0.2,
                                   horizontal_flip= True,
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

#%% Visualizing ImageDataGenerator augmentation
train_datagen_viz = ImageDataGenerator(rotation_range=40,
                                   shear_range=0.1,
                                   zoom_range=0.2,
                                   horizontal_flip= True,
                                   vertical_flip=True,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1)

train_gen_viz = train_datagen_viz.flow_from_directory(
    'train',
    target_size=image_size,
    color_mode='rgb',
    batch_size=1,
    class_mode='binary',
    seed=1337)

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,15))

for i in range(4):
    image = next(train_gen_viz)[0].astype('uint8')
    
    image = np.squeeze(image)
    
    ax[i].imshow(image)
    ax[i].axis('off')
    
#%%
# Units
con_layer_1 = [16]
con_layer_2 = [32]
con_layer_3 = [32]
act_funcs = ['relu', 'selu', 'elu', 'tanh', 'sigmoid']
hidden_layer_1 = [512]
dropout_sizes = [0.2]
optimizers = ["adam"]
epoch_n = 30

# Number of models to be created:
model_count = len(con_layer_1) * len(con_layer_2) * len(con_layer_3) * len(act_funcs) * len(act_funcs) * len(act_funcs) * len(act_funcs) * len(dropout_sizes) * len(optimizers) * len(hidden_layer_1)

# Result array
results = pd.DataFrame(columns = ['loss',
                                  'val_loss',
                                  'accuracy',
                                  'val_accuracy',
                                  'con_layer_1',
                                  'con_layer_1_activation',
                                  'con_layer_2',
                                  'con_layer_2_activation',
                                  'con_layer_3',
                                  'con_layer_3_activation',
                                  'hidden_layer_1',
                                  'hidden_layer_1_activation',
                                  'dropout_sizes',
                                  'optimizers',
                                  'epoch',
                                  'epoch_n'])
print(f"There are {model_count} parameter combinations and {epoch_n} epochs")

# Load results
results = pd.read_csv('loop_results.csv')
#results

# Loop tracking
count = 0 # To count loops
# Model loop
for con1 in con_layer_1:
    for con1_act in act_funcs:
        for con2 in con_layer_2:
            if(con1*3<con2):
                continue
            for con2_act in act_funcs:
                for con3 in con_layer_3:
                    if(con2*3<con3):
                        continue
                    for con3_act in act_funcs:
                        for hidden1 in hidden_layer_1:
                            for hidden1_act in act_funcs:
                                for dropout_size in dropout_sizes:
                                    for optimizer in optimizers:
                                        # Count and print progress
                                        count = count+1
                                        
                                        
                                        # Check if combination has been used
                                        if(len(results.loc[(results['con_layer_1'] == con1) &
                                                           (results['con_layer_1_activation'] <= con1_act) &
                                                           (results['con_layer_2'] == con2) &
                                                           (results['con_layer_2_activation'] == con2_act) &
                                                           (results['con_layer_3'] == con3) &
                                                           (results['con_layer_3_activation'] == con3_act) &
                                                           (results['hidden_layer_1'] == hidden1) &
                                                           (results['hidden_layer_1_activation'] == hidden1_act) &
                                                           (results['dropout_sizes'] == dropout_size) &
                                                           (results['optimizers'] == optimizer) &
                                                           (results['epoch_n'] <= epoch_n)])):
                                            print(f"Skipped model {count} out of {model_count}.")
                                            continue
                                        else:
                                            print(f'Training model {count} out of {model_count}.')
                                        # Model baseret på model.py kopieret 04/11 ~11.30
                                        model = keras.Sequential([
                                            # First convolution
                                            layers.Conv2D(con1, (3,3), activation=con1_act, input_shape=(200, 200, 3)),
                                            layers.MaxPooling2D(2,2), # halving the image size 

                                            # Second convolution
                                            layers.Conv2D(con2, (3,3), activation=con2_act),
                                            layers.MaxPooling2D(2,2),

                                            # Third convolution
                                            layers.Conv2D(con3, (3,3), activation=con3_act),
                                            layers.MaxPooling2D(2,2),

                                            # Fourth convolution - hidden until after some finetuning
                                            #layers.Conv2D(64, (3,3), activation='relu'),
                                            #layers.MaxPooling2D(2,2),

                                            # Fifth convolution - hidden until after some finetuning
                                            #layers.Conv2D(64, (3,3), activation='relu'),
                                            #layers.MaxPooling2D(2,2),

                                            # Flatten results to feed into a Deep Nerual Net
                                            layers.Flatten(),

                                            # 512 neuron hidden layer
                                            layers.Dense(hidden1, activation=hidden1_act),
                                            
                                            # Dropout NEW!!!!!!
                                            layers.Dropout(dropout_size),

                                            # Binary output layer
                                            layers.Dense(1, activation='sigmoid')
                                            ])

                                        #model.summary() # model summary

                                        model.compile(
                                            loss='binary_crossentropy',
                                            optimizer=optimizer,
                                            metrics=['accuracy']) # compiling model
                                        
                                        # Model fitting fra model_2.py kopieret 4/11 ~11.30
                                        # Callbacks for tensorboard 
                                        tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs") # tensorboard --logdir ./logs

                                        # Step sizes for train, validation and testing
                                        STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
                                        STEP_SIZE_VAL=val_gen.n//val_gen.batch_size
                                        STEP_SIZE_TEST=test_gen.n//test_gen.batch_size

                                        # Fitting model
                                        history = model.fit(
                                            train_gen,
                                            steps_per_epoch=STEP_SIZE_TRAIN,
                                            epochs=epoch_n, # for at spare lidt tid
                                            validation_data=val_gen,
                                            validation_steps=STEP_SIZE_VAL,
                                            callbacks=[tensorboard_callback],
                                            verbose = False)
                                        
                                        # Get results (from history)
                                        history = history.history
                                        
                                        # Append each epoch
                                        for h in range(len(history['loss'])):
                                            row = {'loss':history['loss'][h],
                                                   'val_loss':history['val_loss'][h],
                                                   'accuracy':history['accuracy'][h],
                                                   'val_accuracy': history['val_accuracy'][h],
                                                   'con_layer_1': con1,
                                                   'con_layer_1_activation': con1_act,
                                                   'con_layer_2': con2,
                                                   'con_layer_2_activation': con2_act,
                                                   'con_layer_3': con3,
                                                   'con_layer_3_activation': con3_act,
                                                   'hidden_layer_1': hidden1,
                                                   'hidden_layer_1_activation': hidden1_act,
                                                   'dropout_sizes': dropout_size,
                                                   'optimizers': optimizer,
                                                   'epoch': (h+1),
                                                   'epoch_n': epoch_n}
                                            results = results.append(row, ignore_index=True)
                                        results.to_csv('loop_results.csv',index=False)

print("All done")

#%%
# Save model
results.to_csv('loop_results.csv',index=False)