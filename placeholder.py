import pandas
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random
import requests
from PIL import Image
import cv2
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import pickle
import pandas as pd
import os
import imghdr
import zipfile
import requests
from io import StringIO, BytesIO
import matplotlib.image as mpimg
import matplotlib
#import tensorflow-gpu

BATCH_SIZE = 20
NUM_CLASSES = 200
NUM_IMAGES_PER_CLASS = 500
TRAINING_IMAGES_DIR = './tiny-imagenet-200/train/'

NUM_VAL_IMAGES = 10000
VAL_IMAGES_DIR = './tiny-imagenet-200/val/'

IMAGES_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
IMAGES_NUM = 100000
IMAGE_SIZE = 64
NUM_CHANNELS = 3
IMAGE_ARR_SIZE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS

### Preprocess image

# Downloading folder
def download_images(url):
    if (os.path.isdir(TRAINING_IMAGES_DIR)):
        print ('Images have already been downloaded')
        return
    r = requests.get(url, stream=True)
    print ('Downloading ' + url )
    zip_ref = zipfile.ZipFile(BytesIO(r.content))
    zip_ref.extractall('./')
    zip_ref.close()

#loading the images

def loading_image(imagedir, batch_size=500):
    image_index = 0
    images = np.ndarray(shape=(IMAGES_NUM, IMAGE_ARR_SIZE))
    names = []
    labels = []

    for i in os.listdir(imagedir):
        if os.path.isdir(imagedir + i + '/images/'):
            type = os.listdir(imagedir + i + '/images/')
            #Looping through all the images of a type directory
            batch_index = 0
            
            for image in type:
                image_file = os.path.join(imagedir, i + '/images/', image)

                #Reading Images as they are; n
                image_data = mpimg.imread(image_file)
                plt.imshow(image, cmap="gray")
                plt.show()

                
                if (image_data.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)):
                    images[image_index, :] = image_data.flatten()

                    labels.append(i)
                    names.append(image)
                    
                    image_index += 1
                    batch_index += 1
                if (batch_index >= batch_size):
                    break;

    return (images, np.asarray(labels), np.asarray(names))

#displaying the image
def plot_object(data):
    plt.figure(figsize=(1,1))
    image = data.reshape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    plt.imshow(image, cmap = matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    plt.show()



download_images(IMAGES_URL)
#training_images, training_labels, training_files = loading_image(TRAINING_IMAGES_DIR, batch_size=BATCH_SIZE)

for i in os.listdir(TRAINING_IMAGES_DIR):
    type = os.listdir(TRAINING_IMAGES_DIR + i + '/images/')
    #Looping through all the images of a type directory
    batch_index = 0
            
    for image in type:
        image_file = os.path.join(TRAINING_IMAGES_DIR, i + '/images/', image)

        #Reading Images as they are; n
        image_data = mpimg.imread(image_file)
        plot_object(image_data)
        break; #only want to see one image
        

#data_dir = os.listdir(r"C:\Users\madok\Documents\Smart-Tech-CA1\tiny-imagenet-200\val\images")
#print(data_dir)

#img = cv2.imread(os.path.join('tiny-imagenet-200','val','images', 'val_0.jpeg'))
#print(img.shape)
#plt.imshow(img)
#plt.show()