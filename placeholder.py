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
#import tensorflow-gpu

BATCH_SIZE = 20
NUM_CLASSES = 200
NUM_IMAGES_PER_CLASS = 500
TRAINING_IMAGES_DIR = './tiny-imagenet-200/train/'

NUM_VAL_IMAGES = 10000
VAL_IMAGES_DIR = './tiny-imagenet-200/val/'

IMAGES_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

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

download_images(IMAGES_URL)


#data_dir = os.listdir(r"C:\Users\madok\Documents\Smart-Tech-CA1\tiny-imagenet-200\val\images")
#print(data_dir)

#img = cv2.imread(os.path.join('tiny-imagenet-200','val','images', 'val_0.jpeg'))
#print(img.shape)
#plt.imshow(img)
#plt.show()