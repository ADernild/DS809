# -*- coding: utf-8 -*-
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from numpy.random import seed # to set seed
from tensorflow.random import set_seed # to set seed
import pandas as pd
import os
import tensorflow as tf
from tensorflow.python.client import device_lib

# Set path to this file location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


# Apply seed
seed_value = 1338
set_seed(seed_value)
seed(seed_value)

#%% ImageDataGenerator
image_size = [200, 200]
batch = 32

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

# Units
con_layer_1 = [16]
con_layer_2 = [32]
con_layer_3 = [32, 64]
con_layer_4 = [32, 64, 128]
con_layer_5 = [32, 64, 128]
act_funcs_con = ['relu']
act_funcs_con_last = ['selu', 'elu', 'tanh']
act_funcs_hidden = ['sigmoid', 'relu']
hidden_layer_1 = [512, 1024]
dropout_sizes = [0.2, 0.5]
optimizers = ["adam"]
epoch_n = 30

# Number of models to be created:
model_count = len(con_layer_1) * len(con_layer_2) * len(con_layer_3) * len(con_layer_4) * len(con_layer_5) * len(act_funcs_con) * len(act_funcs_con) * len(act_funcs_con_last) * len(dropout_sizes) * len(optimizers) * len(hidden_layer_1) * len(act_funcs_hidden)

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
                                  'con_layer_4',
                                  'con_layer_4_activation',
                                  'con_layer_5',
                                  'con_layer_5_activation',
                                  'hidden_layer_1',
                                  'hidden_layer_1_activation',
                                  'dropout_sizes',
                                  'optimizers',
                                  'epoch',
                                  'epoch_n'])
print(f"There are {model_count} parameter combinations and {epoch_n} epochs")

# Loop tracking
count = 0 # To count loops

# Model loop
for con1 in con_layer_1:
    for con1_act in act_funcs_con:
        for con2 in con_layer_2:
            if(con1*3<con2):
                count = count+1
                print(f"Skipped model {count} out of {model_count}, bc. kernel difference from convolution 1 to 2.")
                continue
            for con2_act in act_funcs_con:
                for con3 in con_layer_3:
                    if(con2*3<con3):
                        count = count+1
                        print(f"Skipped model {count} out of {model_count}, bc. kernel difference from convolution 2 to 3.")
                        continue
                    for con3_act in act_funcs_con:
                        for con4 in con_layer_4:
                            if(con3*3<con4):
                                count = count+1
                                print(f"Skipped model {count} out of {model_count}, bc. kernel difference from convolution 3 to 4.")
                                continue
                            for con4_act in act_funcs_con:
                                for con5 in con_layer_5:
                                    if(con4*3<con5):
                                        count = count+1
                                        print(f"Skipped model {count} out of {model_count}, bc. kernel difference from convolution 4 to 5.")
                                        continue
                                    for con5_act in act_funcs_con_last:                        
                                        for hidden1 in hidden_layer_1:
                                            for hidden1_act in act_funcs_hidden:
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
                                                                           (results['con_layer_4'] == con4) &
                                                                           (results['con_layer_4_activation'] == con4_act) &
                                                                           (results['con_layer_5'] == con5) &
                                                                           (results['con_layer_5_activation'] == con5_act) &
                                                                           (results['hidden_layer_1'] == hidden1) &
                                                                           (results['hidden_layer_1_activation'] == hidden1_act) &
                                                                           (results['dropout_sizes'] == dropout_size) &
                                                                           (results['optimizers'] == optimizer) &
                                                                           (results['epoch_n'] <= epoch_n)])):
                                                            print(f"Skipped model {count} out of {model_count}, bc. redundancy.")
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

                                                            # Dropout
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
                                                                   'con_layer_4': con4,
                                                                   'con_layer_4_activation': con4_act,
                                                                   'con_layer_5': con5,
                                                                   'con_layer_5_activation': con5_act,
                                                                   'hidden_layer_1': hidden1,
                                                                   'hidden_layer_1_activation': hidden1_act,
                                                                   'dropout_sizes': dropout_size,
                                                                   'optimizers': optimizer,
                                                                   'epoch': (h+1),
                                                                   'epoch_n': epoch_n}
                                                            results = results.append(row, ignore_index=True)
                                                        # Save model
                                                        results.to_csv('loop_results.csv',index=False)

print("All done")