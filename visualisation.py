import sys

import matplotlib as plt

loss_train = history.history['train_loss']
loss_val = history.history['val_loss']
epochs = range(1,35)

plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show() #or is it plt.print() ???

loss_train = history.history['accuracy']
loss_val = history.history['val_accuracy']
epochs = range(1,11)

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

def learning_summary(history):
    #Plotting loss
    plt.subplot(210)
    plt.title('Cross entropy loss')
    plt.plot(history.history['loss'], color = 'red', label = 'training')
    plt.plot(history.history['validation_loss'], color = 'blue', label = 'test')
    #Plotting the accuracy
    plt.subplot(211)
    plt.title('Accuracy of classification')
    plt.plot(history.history['accuracy'], color = 'orange', label = 'training')
    plt.plot(history.history['validation_accuracy'], color = 'green', label = 'test')
    #Plot saved to file
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + 'plt.png')
    plt.close()