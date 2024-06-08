import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
import prelu

# Function for draw bounding boxes on images
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

model_path = "pnet_model"
# Define custom objects dictionary with custom layers
custom_objects = {'PReLU': prelu.PReLU}

# Load the model with custom objects
loaded_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)


# Function to process a single image
# Function to process a single image
# # Function to process a single image
def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Resize the image to match the expected input shape of the model
    resized_image = cv2.resize(image, (160, 160))

    # Convert the image to float32 and normalize
    resized_image = resized_image.astype('float32') / 255.0

    # Make predictions on the image
    predictions = loaded_model.predict(np.expand_dims(resized_image, axis=0))

    # Extract bounding box predictions
    bbox_pred = predictions['bbox_output'][0]  # Assuming single image prediction
    class_pred = predictions['class_output'][0]  # Assuming single image prediction

    # Draw bounding box on the image
    image_with_bbox_pred = draw_bounding_box(image, bbox_pred)

    # Display the image with predictions
    plt.imshow(image_with_bbox_pred)
    plt.title(f'Predicted Bbox: {bbox_pred}, Predicted Class: {class_pred}')
    plt.show()

# Example usage:
image_path = "new/images/aalu.jpg"
process_image(image_path)


