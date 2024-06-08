import cv2
import numpy as np
from scipy.spatial.distance import cosine
import tensorflow as tf
import prelu

# Load the FaceNet model
facenet_model = tf.saved_model.load('facenet_saved_model')

model_path = "pnet_model"
# Define custom objects dictionary with custom layers
custom_objects = {'PReLU': prelu.PReLU}

# Load the model with custom objects
loaded_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# Function to preprocess an image
def preprocess_image(image):
    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize image to a standard size
    resized = cv2.resize(image_rgb, (160, 160))
    # Convert the image to float32
    resized_float32 = resized.astype('float32') / 255.0
    return resized_float32

# Function to classify a face using FaceNet embeddings and cosine similarity
def classify_face(image, known_embeddings, threshold=0.5):
    # Extract embedding from the image using FaceNet
    embedding = extract_embedding(image)
    if embedding is not None:
        # Calculate cosine similarity between the embedding and known embeddings
        similarities = [1 - cosine(embedding, known_embedding) for known_embedding in known_embeddings]
        # Get the index of the highest similarity
        max_index = np.argmax(similarities)
        # Check if the similarity is above the threshold
        if similarities[max_index] > threshold:
            # Return the index corresponding to the highest similarity
            return max_index
        else:
            # If similarity is below threshold, return None
            return None
    else:
        return None

# Function to extract embeddings using FaceNet
# Function to extract embeddings using FaceNet
def extract_embedding(image):
    # Use the loaded_model for face detection
    faces = loaded_model.predict(np.expand_dims(image, axis=0))
    # print(f'hellooooooooooooooo- {faces['bbox_output']}')
    # if len(faces) > 0 and faces[0] is not None:
    # Assuming only one face is detected, consider the first one
    face = faces['bbox_output'][0]
    # Assuming the output of the model is [x, y, width, height]
    x, y, width, height = face
    # Crop face from the image
    face_image = image[int(y):int(y+height), int(x):int(x+width)]
    # Preprocess the face for FaceNet
    preprocessed_face = preprocess_image(face_image)
    # Get embedding using FaceNet
    embedding = get_embedding(facenet_model, preprocessed_face)
    return embedding
    # else:
    #     return None


# Function to get embedding using FaceNet model
def get_embedding(model, face):
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    sample = np.expand_dims(face, axis=0)
    # Convert the input to a TensorFlow tensor
    input_tensor = tf.convert_to_tensor(sample)
    # Perform inference using the model
    embedding = model.signatures["serving_default"](input_tensor)
    # Convert the embedding to a numpy array
    embedding = embedding['Bottleneck_BatchNorm'].numpy()
    return embedding[0]

# Function to draw bounding box and label
def draw_box(image, box, label):
    x, y, w, h = box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

# Load known embeddings and labels
known_data = np.load("train_data_with_facenet.npz")
known_embeddings = known_data['embeddings']
known_labels = known_data['labels']

# Load the input image
input_image = cv2.imread("new/images/h.jpg")

# Preprocess the input image
preprocessed_image = preprocess_image(input_image)

recognized_names = []  # To store recognized names

# Extract embeddings and classify faces in the input image
#label_index = classify_face(preprocessed_image, known_embeddings)
# if label_index is not None:
#     # Get the label corresponding to the index
#     label = known_labels[label_index]
#     recognized_names.append(label)
#     # Draw bounding box and label
#     draw_box(input_image, (0, 0, preprocessed_image.shape[1], preprocessed_image.shape[0]), label)

# Show the result
cv2.imshow('Face Recognition Result', input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print recognized names
print("Recognized names:")
for name in recognized_names:
    print(name)
