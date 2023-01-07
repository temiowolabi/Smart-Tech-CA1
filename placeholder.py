import numpy as np
import cv2
import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import requests
from io import BytesIO

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

    index = 0
    name_index = {}
    for name in names:
        folder = DIR + "train/" + name + "/images/"
       # print(folder)
        for img in os.listdir(TRAINING_IMAGES_DIR + name + "/images/"):
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


# arr1 = np.array([1, 2, 3])
# arr2 = np.array([4, 5, 6, 7])
#
# arr2 = np.resize(arr2, arr1.shape)
# mean = np.mean([arr1, arr2])


def gray_scale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalise(img):
    img = cv2.equalizeHist(img)
    return img


def preprocess(img):
    img = gray_scale(img)
    img = equalise(img)
    img = img / 255
    return img


download_images(IMAGES_URL)
# names, labels, x, y, h, w = validation_data()
# names = training_data()
# training_data()
# print(names)
# test1, test2 = test_data()

# plt.imshow(test1[0])

# plt.title(test2[0])
# plt.xlabel("X-axis")
# plt.ylabel("Y-axix")

# plt.figure(figsize=(5,5))
# plt.show()


# DATA EXPLORATION

# Load the training data
images, labels, bounding_boxes = training_data()

# Visualise some of the images and their labels
# n_images = 3  # Number of images to display
# for i in range(n_images):
#     plt.imshow(images[i])
#     plt.title(labels[i])
#     plt.show()

# Iterating over images and displaying a label for each image

# Calculate summary statistics for the bounding boxes
bounding_box_values = [box for box in bounding_boxes if box is not None]
min_bounding_box = np.min(bounding_box_values)  # Calculate min bounding box
max_bounding_box = np.max(bounding_box_values)  # Calculate max bounding box
print("Minimum bounding box value:", min_bounding_box)
print("Maximum bounding box value:", max_bounding_box)

# Print the number of images and labels
print("Number of images:", len(images))
print("Number of labels:", len(labels))