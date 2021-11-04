import sys
import model_2
import matplotlib as plt

#History is changed to hist based on the .py file model_2
#If model_2 is used as is, this .py file should have the proper imports from model_2

loss_train = hist.hist['train_loss']
loss_val = hist.hist['val_loss']
epochs = range(1,35)

plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show() #or is it plt.print() ???

loss_train = hist.hist['accuracy']
loss_val = hist.hist['val_accuracy']
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