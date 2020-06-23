
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.python.lib.io import file_io
from skimage.transform import rescale, resize

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Function that reads the data from the csv file, increases the size of the images and returns the images and their labels
# dataset: Data path
IMAGE_SIZE = 48

# Function that reads the data from the csv file, increases the size of the images and returns the images and their labels
# dataset: Data path
IMAGE_SIZE = 48

def get_data(dataset, small=False, size=2000):
    
    file_stream = file_io.FileIO(dataset, mode='r')
    data = pd.read_csv(file_stream)
    data[' pixels'] = data[' pixels'].apply(lambda x: [int(pixel) for pixel in x.split()])
    X, Y = data[' pixels'].tolist(), data['emotion'].values
    X = np.array(X, dtype='float32').reshape(-1,IMAGE_SIZE, IMAGE_SIZE,1)
    X = X/255.0
    
    if small==True: 
        X = X[0:size,:,:,:]
        Y = Y[0:size]
        
    X_res = np.zeros((X.shape[0], Resize_pixelsize,Resize_pixelsize,3))
    for ind in range(X.shape[0]): 
        sample = X[ind]
        sample = sample.reshape(IMAGE_SIZE, IMAGE_SIZE)
        image_resized = resize(sample, (Resize_pixelsize, Resize_pixelsize), anti_aliasing=True)
        X_res[ind,:,:,:] = image_resized.reshape(Resize_pixelsize,Resize_pixelsize,1)

    Y_res = np.zeros((Y.size, 7))
    Y_res[np.arange(Y.size),Y] = 1    
    
    return  X, X_res, Y_res

Resize_pixelsize = 197

dev_dataset_dir = '../data/raw/fer_csv/dev.csv'
test_dataset_dir = '../data/raw/fer_csv/test.csv'

X_dev, X_res_dev, Y_dev   = get_data(dev_dataset_dir,small=True, size=1000)
X_test, X_res_test, Y_test   = get_data(test_dataset_dir, small=True, size=1000)


Resnet_model = tf.keras.models.load_model('../models/tl/ResNet-BEST-73.2.h5')


print('\n# Evaluate on dev data')
results_dev = Resnet_model.evaluate(X_res_dev,Y_dev)
print('dev loss, dev acc:', results_dev)

print('\n# Evaluate on test data')
results_test = Resnet_model.evaluate(X_res_test,Y_test)
print('test loss, test acc:', results_test)
