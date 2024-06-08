import cv2
import numpy as np
from mtcnn import MTCNN
from keras.models import load_model
from scipy.spatial.distance import cosine
import tensorflow as tf
import prelu

# Load the FaceNet model
facenet_model = tf.saved_model.load('facenet_saved_model')

# Initialize MTCNN for face detection
#detector = MTCNN()
model_path = "pnet_model"
# Define custom objects dictionary with custom layers
custom_objects = {'PReLU': prelu.PReLU}

# Load the model with custom objects
detector = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
# Function to preprocess an image
def preprocess_image(image):
    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize image to a standard size
    resized = cv2.resize(image_rgb, (160, 160))
    return resized

# Function to classify a face using FaceNet embeddings and cosine similarity
def classify_face(image, known_embeddings, threshold=0.5):
    # Extract embedding from the image using MTCNN and FaceNet
    embedding = extract_embedding(image)
    if embedding is not None:
        # Calculate cosine similarity between the embedding and known embeddings
        similarities = [1 - cosine(embedding, known_embedding) for known_embedding in known_embeddings]
        # Get the index of the highest similarity
        max_index = np.argmax(similarities)
        # Check if the similarity is above the threshold
        if similarities[max_index] > threshold:
            # Return the label corresponding to the highest similarity
            return max_index
        else:
            # If similarity is below threshold, return None
            return None
    else:
        return None

# Function to extract embeddings using FaceNet
def extract_embedding(image):
    # Detect faces in the image using MTCNN
    faces = detector.detect_faces(image)
    if faces:
        # Get the first face detected (assuming only one face in the image)
        x, y, w, h = faces[0]['box']
        # Crop face from the image
        face = image[y:y+h, x:x+w]
        # Preprocess the face for FaceNet
        preprocessed_face = preprocess_image(face)
        # Get embedding using FaceNet
        embedding = get_embedding(facenet_model, preprocessed_face)
        return embedding
    else:
        return None

# Function to get embedding using FaceNet model
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

# Real-time face detection and classification
cap = cv2.VideoCapture(0)

label_to_store = None  # Initialize variable to store the label

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect faces in the frame
    faces = detector.detect_faces(frame)
    
    # Only process the first face detected
    if faces:
        face = faces[0]  # Get the first face detected
        x, y, w, h = face['box']
        # Preprocess the face for classification
        preprocessed_face = preprocess_image(frame[y:y+h, x:x+w])
        # Classify the face
        label_index = classify_face(preprocessed_face, known_embeddings)
        if label_index is not None:
            # Get the label corresponding to the index
            label = known_labels[label_index]
            # Draw bounding box and label
            draw_box(frame, (x, y, w, h), label)
    
    # Show the label to store on the screen
    if label_to_store is not None:
        cv2.putText(frame, f"Press 's' to store label: {label_to_store}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    cv2.imshow('Real-time Face Classification', frame)
    
    # Check for key press events
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        if label_to_store is not None:
            print("Label stored:", label_to_store)
            # Here you can store the label_to_store in your desired way (e.g., in a list)
            label_to_store = None
        else:
            # Get the label of the last classified face and store it
            if faces:
                face = faces[0]  # Get the first face detected
                x, y, w, h = face['box']
                preprocessed_face = preprocess_image(frame[y:y+h, x:x+w])
                label_index = classify_face(preprocessed_face, known_embeddings)
                if label_index is not None:
                    label_to_store = known_labels[label_index]

cap.release()
cv2.destroyAllWindows()