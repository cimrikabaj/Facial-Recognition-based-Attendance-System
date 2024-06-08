import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 # opencv
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot as plt
from keras.models import load_model
from PIL import Image
import os
import tensorflow as tf
import prelu
import json
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# facenet_model = load_model('facenet_keras.h5')
# print('Loaded Model')
model_path="pnet_model"
# Define custom objects dictionary with custom layers
custom_objects = {'PReLU': prelu.PReLU}

# Load the model with custom objects
loaded_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
# loaded_model = load_model('pnet_model.h5')

# img = cv2.imread('extra_stuffs/data/kan076bct004/2.jpg')
# plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
# #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()
# print(img.shape)

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def load_and_preprocess_data(image_paths):
    images = []

    # for image_path in image_paths:
    # if image_paths.endswith(".jpg"):
            # Load and preprocess image
    image = load_img(image_paths, target_size=(128, 128))
    image = img_to_array(image) / 255.0

    images.append(image)

    return images


def draw_bounding_box(image, bbox):
    image_copy = np.copy(image)
    h, w, _ = image_copy.shape

    x, y, width, height = bbox
    x = int(x * w)
    y = int(y * h)
    width = int(width * w)
    height = int(height * h)

    cv2.rectangle(image_copy, (x, y), (x + width, y + height), (0, 255, 0), 2)

    return image_copy

image_folder = 'new/dhukha/87ea2e4b-a159-11ee-a0ff-965c5d65b1d8.1.JPG'
train = load_and_preprocess_data(image_folder)
train_images = tf.convert_to_tensor(train, dtype=tf.float32)
train_dataset = tf.data.Dataset.from_tensor_slices((train_images))
BATCH_SIZE=1
train_dataset = train_dataset.shuffle(len(train_images)).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

print(train)

# for images in train_dataset:
#     # Get predictions from the model
#     predictions = loaded_model.predict(images)

#     # Directly extract bounding box predictions
#     bbox_predictions = predictions
#     # Since you're not using class predictions, you can remove this line:
#     # class_predictions = predictions['class_output']

#     # Select a random sample from the batch
#     idx = np.random.randint(0, len(images))
#     image = images[idx].numpy()
#     bbox_pred = bbox_predictions[idx]  # No need for .numpy()
#     # Remove the line for class prediction since it's not used in the code
#     # class_pred = class_predictions[idx]  # No need for .numpy()

#     # Draw bounding box on the image
#     image_with_bbox_pred = draw_bounding_box(image, bbox_pred)

#     # Display the images
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 2)
#     plt.imshow(image_with_bbox_pred)
#     # Remove class prediction from the title since it's not used
#     plt.title(f'Predicted Bbox: {bbox_pred}')

#     plt.show()


for images in train_dataset:
    # Get predictions from the model
    predictions = loaded_model.predict(images)

    # Directly extract bounding box predictions
    bbox_predictions = predictions['bbox_output']
    # Since you're not using class predictions, you can remove the line for class_predictions

    # Select a random sample from the batch
    idx = np.random.randint(0, len(images))
    image = images[idx].numpy()
    bbox_pred = bbox_predictions[idx]  # No need for .numpy()

    # Draw bounding box on the image
    image_with_bbox_pred = draw_bounding_box(image, bbox_pred)

    # Display the images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 2)
    plt.imshow(image_with_bbox_pred)
    # Remove class prediction from the title since it's not used
    plt.title(f'Predicted Bbox: {bbox_pred}')

    plt.show()

