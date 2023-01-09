from glob import glob

import numpy as np
import cv2
import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import requests
from io import BytesIO
from collections import Counter
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf

BATCH_SIZE = 20
NUM_CLASSES = 200
NUM_IMAGES_PER_CLASS = 500
NUM_VAL_IMAGES = 10000
DIR = './tiny-imagenet-200/'
TEST_IMAGES_DATA = './tiny-imagenet-200/test'
TRAINING_IMAGES_DIR = './tiny-imagenet-200/train/'
VAL_IMAGES_DIR = './tiny-imagenet-200/val/'
VALIDATION_ANNOTATIONS_FILE = 'val_annotations.txt'
WORDNETID = 'wnids.txt'
WORDS = 'words.txt'
IMAGES_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'


### Preprocess image

# Downloading folder
def download_images(url):
    if (os.path.isdir(TRAINING_IMAGES_DIR)):
        print('Images have already been downloaded')
        return

    r = requests.get(url, stream=True)
    print('Downloading ' + url)
    zip_ref = zipfile.ZipFile(BytesIO(r.content))
    zip_ref.extractall('./')
    zip_ref.close()


def validation_data():
    val_images = []
    val_labels = []
    val_x = []
    val_y = []
    val_h = []
    val_w = []

    with open(VAL_IMAGES_DIR + VALIDATION_ANNOTATIONS_FILE) as f:
        text = f.readlines()
        for i in text:
            i_items = i.strip().split('\t')
            images, labels, x, y, h, w = i_items
            val_images.append(images)
            val_labels.append(labels)
            val_x.append(x)
            val_y.append(y)
            val_h.append(h)
            val_w.append(w)
    return val_images, val_labels, val_x, val_y, val_h, val_w


def training_data():
    images = []
    labels = []

    with open(DIR + WORDNETID) as file:
        names = [label.strip() for label in file.readlines()]

    # index = 0
    # name_index = {}
    # for name in names:
    #     folder = DIR + "train/" + name + "/images/"
    #    # print(folder)
    #     for img in os.listdir(TRAINING_IMAGES_DIR + name + "/images/"):
    #         images.append(mpimg.imread(folder + img))
    #         label_list.append(name)
    #         name_index[img] = index
    #         index += 1

    index = 0
    name_index = {}
    for name in names:
        folder = DIR + "train/" + name + "/images/"
        for img in os.listdir(folder):
            images.append(mpimg.imread(folder + img))
            labels.append(name)
            name_index[img] = index
            index += 1

    bounding_box = [None for _ in range(len(images))]
    for name in names:
        box = TRAINING_IMAGES_DIR + name + "/" + name + "_boxes.txt"
        with open(box) as file2:
            text = file2.readlines()
            for i in text:
                i_items = i.strip().split()
                bounding_box[name_index[i_items[0]]] = list(map(int, i_items[1:]))
    return images, labels, bounding_box


# X_train, y_train, bounding_box = training_data()
#
# # Get the shape of the training data
# num_images, height, width, num_channels = X_train.shape
# print("Number of images:", num_images)
# print("Image size:", (height, width))
# print("Number of channels:", num_channels) # This is referring to colour channels

def labels():
    with open(DIR + WORDNETID) as file:
        labels = [label.strip() for label in file.readlines()]
    index = {label: index for index, label in enumerate(labels)}

    return index


def test_data():
    test_img = []
    index = []

    folder = TEST_IMAGES_DATA + "/images/"
    for file in os.listdir(folder):
        image = mpimg.imread(folder + file)
        test_img.append(image)
        index.append(file)
    return test_img, index

def resize_images(input_dir, output_dir, size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        # Load the image
        image = cv2.imread(os.path.join(input_dir, filename))

        # Resize the image
        image = cv2.resize(image, size)

        # Save the resized image
        cv2.imwrite(os.path.join(output_dir, filename), image)


def normalize_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        # Load the image as a NumPy array
        image = np.load(os.path.join(input_dir, filename))

        # Normalize the pixel values
        mean = np.mean(image)
        std = np.std(image)
        image = (image - mean) / std

        # Save the normalized image
        np.save(os.path.join(output_dir, filename), image)



def gray_scale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalise(img):
    img = cv2.equalizeHist(img)
    return img


def preprocess(img):
    img = gray_scale(img)
    img = equalise(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)  # apply gaussian blur
    img = img/255
    return img

download_images(IMAGES_URL)


X_train, y_train = np.array(training_data())

#ßX_train = np.array(X_train)
#X_train = preprocess_data(X_train)


labels = set(y_train)
label_to_integer = {label: index for index, label in enumerate(labels)}

def one_hot_encode(labels):
    # Convert the labels to integers
    integer_labels = [label_to_integer[label] for label in labels]
    # One-hot encode the labels
    one_hot_labels = keras.utils.to_categorical(integer_labels, num_classes=NUM_CLASSES)
    return one_hot_labels


y_train = one_hot_encode(y_train)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# Preprocess the validation data
X_val, y_val = np.array(validation_data())


#X_val = preprocess_data(X_val)
y_val = one_hot_encode(y_val)
print(len(X_val))
print(len(y_val))

# Load the testing data
images, label_list = test_data()
def display_images():
    # Visualise some of the images and their label_list
    n_images = 3  # Number of images to display
    for i in range(n_images):
        plt.imshow(images[i])
        #plt.title(label_list[i])
        height, width = images[i].shape[:2]  # Retrieve image size
        plt.text(x=0, y=0, s=f'{label_list[i]}: {height}x{width}')
        plt.show()


def count_image_in_each_class():
    # Count the number of images in each class
    class_counts = Counter(label_list)

    # Print the class counts
    print(class_counts)

    # Display a histogram of the class counts
    plt.hist(class_counts.values())
    plt.show()



def display_image_by_size():
    # Extract the width and height of the images
    widths = [image.shape[1] for image in images]
    heights = [image.shape[0] for image in images]

    # Plot the widths and heights on a scatter plot
    plt.scatter(widths, heights)
    plt.show()


#Define the model
model = tf.keras.Sequential()
# Add a convolutional layer with 32 filters, a 3x3 kernel, and padding set to 'same'
model.add(tf.keras.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(64, 64, 3)))
# Add a max pooling layer with a 2x2 pool size
model.add(tf.keras.MaxPooling2D(pool_size=(2, 2)))
# Add a flatten layer
model.add(tf.keras.Flatten())
# Add a dense layer with 128 units and ReLU activation
model.add(tf.keras.Dense(128, activation='relu'))
# Add a dense layer with NUM_CLASSES units and softmax activation
model.add(tf.keras.Dense(NUM_CLASSES, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64)

val_loss, val_acc = model.evaluate(X_val, y_val)
print('Validation loss:', val_loss)
print('Validation accuracy:', val_acc)

