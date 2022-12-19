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
from sklearn import preprocessing
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

def trainingImages(dir):
    images = []
    for i in os.listdir(dir):
        if os.path.isdir(dir + i + '/images/'):
            path = os.listdir(dir + i + '/images/')

            for image in path:
                img_arr = cv2.imread(os.path.join(dir + i + '/images/', image), cv2.IMREAD_GRAYSCALE)
                plt.imshow(img_arr, cmap="gray")
                plt.show()





download_images(IMAGES_URL)
trainingImages(TRAINING_IMAGES_DIR)







