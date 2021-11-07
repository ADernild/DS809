import keras
from keras import models, Model
from matplotlib import pyplot
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot as plt
from numpy import expand_dims
import sys




model = keras.models.load_model('aid/final_model_2.h5')
model.summary()

for layer in model.layers:
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# get filter weights
	filters, biases = layer.get_weights()
	print(layer.name, filters.shape)


filters, biases = model.layers[0].get_weights()
# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# plot first few filters
n_filters, ix = 6, 1
for i in range(n_filters):
    # get the filter
    f = filters[:, :, :, i]
    # plot each channel separately
    for j in range(3):
        # specify subplot and turn of axis
        ax = pyplot.subplot(n_filters, 3, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        pyplot.imshow(f[:, :, j], cmap='brg')
        ix += 1
	# show the figure
filename = sys.argv[0].split('/')[-1]
plt.savefig(filename + 'filter_plt_brg.png')
plt.close()
pyplot.show()

for i in range(len(model.layers)):
	layer = model.layers[i]
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# summarize output shape
	print(i, layer.name, layer.output.shape)


model.summary()
model = Model(inputs=model.inputs, outputs=model.layers[0].output)

img = load_img('train/dog/dog.1.jpg', target_size=(200, 200))
# convert the image to an array
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
# get feature map for first hidden layer
featuremaps = model.predict(img)
# plot all 64 maps in an 8x8 squares

square = 4
ix = 1

for _ in range(square):
    for _ in range(square):
        # specify subplot and turn of axis
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        plt.imshow(featuremaps[0, :, :, ix-1], cmap='brg')
        ix += 1
	# show the figure
filename = sys.argv[0].split('/')[-1]
plt.savefig(filename + 'map_plt_brg.png')
plt.close()
plt.show()



