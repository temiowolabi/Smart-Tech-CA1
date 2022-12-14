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
#import tensorflow-gpu


### Preprocess image

#data_dir = os.listdir(r"C:\Users\madok\Documents\Smart-Tech-CA1\tiny-imagenet-200\val\images")
#print(data_dir)

img = cv2.imread(os.path.join('tiny-imagenet-200','val','images', 'val_0.jpeg'))
print(img.shape)
plt.imshow(img)
plt.show()