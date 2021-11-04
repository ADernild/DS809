import keras
import os
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator
from numpy import asarray
from keras import layers, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import glob
import pandas as pd
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

#Issue; Unsure if every image in /train/cat and /train/dog is "imported", or if it's just pointing on /train and the two folders(?)
#brain not working atm to figure this out or fix it


filenames = os.listdir("./train") #<---

categories=[] #Empty

for f_name in filenames:
    category = f_name.split('.')[0]
    if category=='dog':
        categories.append(1)
    else:
        categories.append(0)

df=pd.DataFrame({'filename': filenames, 'category':categories}) #Dataframe (df)

#Train your stuff with patience, lol, otherwise it may burn down your house
earlystop = EarlyStopping(patience=10)
learning_rate_red = ReduceLROnPlateau(monitor='val_acc', patience=5, verbose=1, factor=0.5, min_lr=0.00001)
callbacks = [earlystop, learning_rate_red]

#Df from earlier / NEEDS TO BE FIXED / TO DO
df["category"] = df["category"].replace({0:'cat',1:'dog'})
training_df, validate_df=train_test_split(df, test_size=0.20, random_state=42)
training_df = training_df.reset_index(drop=True) #Unsure if needed - reset on index + drop
validate_df=validate_df.reset_index(drop=True)

imagex = 200
imagey = 200
image_size = [imagex + imagey]
batch = 30

train_data_gen = ImageDataGenerator(rotation_range=40, rescale=1./255, shear_range=0.1, zoom_range=0.2, horizontal_flip= False, vertical_flip=True, width_shift_range=0.1, height_shift_range=0.1,)
train_gen = train_data_gen.flow_from_dataframe(training_df, "./train", x_col='filename', y_col='category', target_size=image_size, class_mode='categorical', batch_size=batch)


model = keras.Sequential()
        #Layers, Pooling, Dropout and Density
layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(image_size, 3)),
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
model.fit(image_size, batch_size=30, epochs=100)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

history = model.fit_generator(train_gen, steps_per_epoch=len(train_gen)),

# Needed: History on model ex. history = model.fit(x,y, epochs= ,batch_size= , val= etc)

# Needed: Plots/ visuals to show the training i.e. to be implemented

