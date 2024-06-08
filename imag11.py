import cv2
import numpy as np
from scipy.spatial.distance import cosine
import tensorflow as tf
import joblib

#FaceNet model
facenet_model = tf.saved_model.load('facenet_saved_model')

#MTCNN 
detector = joblib.load('mtcnn_model.joblib')

def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (160, 160))
    return resized


def classify_face(image, known_embeddings, threshold=0.5):
    embedding = extract_embedding(image)
    if embedding is not None:
        similarities = [1 - cosine(embedding, known_embedding) for known_embedding in known_embeddings]
        max_index = np.argmax(similarities)
        if similarities[max_index] > threshold:
            return max_index
        else:
            return None
    else:
        return None


def extract_embedding(image):
    faces = detector.detect_faces(image)
    if faces:
        x, y, w, h = faces[0]['box']
        face = image[y:y+h, x:x+w]
        preprocessed_face = preprocess_image(face)
        embedding = get_embedding(facenet_model, preprocessed_face)
        return embedding
    else:
        return None


def get_embedding(model, face):
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    sample = np.expand_dims(face, axis=0)
    input_tensor = tf.convert_to_tensor(sample)
    embedding = model.signatures["serving_default"](input_tensor)
    embedding = embedding['Bottleneck_BatchNorm'].numpy()
    return embedding[0]


def draw_box(image, box, label):
    x, y, w, h = box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


known_data = np.load("train_data_with_facenet.npz")
known_embeddings = known_data['embeddings']
known_labels = known_data['labels']


input_image = cv2.imread("new/dhukha/6.jpg")


faces = detector.detect_faces(input_image)

recognized_names = []  

for face in faces:
    x, y, w, h = face['box']
    preprocessed_face = preprocess_image(input_image[y:y+h, x:x+w])
    label_index = classify_face(preprocessed_face, known_embeddings)
    if label_index is not None:
        label = known_labels[label_index]
        print(label)
        recognized_names.append(label)
        draw_box(input_image, (x, y, w, h),label)


cv2.imshow('Face Recognition Result', input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


import pandas as pd
from datetime import date
from datetime import datetime

# Read the CSV file containing student information
student_data = pd.read_csv("crnAndName.csv")

# Initialize a dictionary to store the count of attendance and dates
attendance_info = {}

# Define a function to mark attendance
def mark_attendance(student_name):
    if student_name in student_data['CRN'].values:
        # Increment the attendance count
        attendance_info[student_name] = attendance_info.get(student_name, 0) + 1
        # Increment the existing attendance count in the DataFrame
        student_attendance = student_data.loc[student_data['CRN'] == student_name, 'Attendance']
        if pd.isnull(student_attendance).any():
            # If attendance is NaN (i.e., the student hasn't attended before), initialize it to 1
            student_data.loc[student_data['CRN'] == student_name, 'Attendance'] = 1
        else:
            # Otherwise, increment the existing attendance count by 1
            student_data.loc[student_data['CRN'] == student_name, 'Attendance'] += 1
        # Append the current date to the dates attended
        # current_date = str(date.today())
        current_datetime = datetime.now().strftime('%Y-%m-%d(%H:%M:%S)')
        if 'Dates Attended' in student_data.columns:
            # Check if the student has attended before
            if pd.isnull(student_data.loc[student_data['CRN'] == student_name, 'Dates Attended']).any():
                student_data.loc[student_data['CRN'] == student_name, 'Dates Attended'] = current_datetime
            else:
                student_data.loc[student_data['CRN'] == student_name, 'Dates Attended'] += ', ' + current_datetime
        else:
            student_data['Dates Attended'] = ''
            student_data.loc[student_data['CRN'] == student_name, 'Dates Attended'] = current_datetime




# Iterate over recognized names and mark attendance  
for name in recognized_names:
    mark_attendance(name)

# Save the updated attendance information back to the CSV file
student_data.to_csv("crnAndName.csv", index=False)

# Print recognized students and their attendance count
print("Recognized students and their attendance count:")
for student, count in attendance_info.items():
    print(f"{student}: {count} times attended")




