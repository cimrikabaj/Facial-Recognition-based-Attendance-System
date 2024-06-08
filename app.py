from flask import Flask, render_template, url_for, redirect,flash,request
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import cv2
import numpy as np
from mtcnn import MTCNN
from keras.models import load_model
from scipy.spatial.distance import cosine
import tensorflow as tf
import pandas as pd
from datetime import date
from datetime import datetime
# Define a function to mark attendance



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:cim123@localhost/users'
app.config['SECRET_KEY'] = 'Ogodoremmanuel'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Load the FaceNet model
facenet_model = tf.saved_model.load('facenet_saved_model')

# Initialize MTCNN for face detection
detector = MTCNN()

# Load known embeddings and labels
known_data = np.load("train_data_with_facenet.npz")
known_embeddings = known_data['embeddings']
known_labels = known_data['labels']


attendance_info = {}


subjects = [
    {"code": "IS", "name": "Information System"},
    {"code": "IAI", "name": "Internet & Intranet"},
    {"code": "EPP", "name": "Engineering Professional Practice"},
    {"code": "SAM", "name": "Simulation and Modeling"},
    {"code": "BD", "name": "Big Data"},
    {"code": "MM", "name": "MultiMedia"}
]

# Function to preprocess an image
def preprocess_image(image):
    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize image to a standard size
    resized = cv2.resize(image_rgb, (160, 160))
    return resized

# Function to classify a face using FaceNet embeddings and cosine similarity
def classify_face(image, threshold=0.5):
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
            return known_labels[max_index]
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
        embedding = get_embedding(preprocessed_face)
        return embedding
    else:
        return None

# Function to get embedding using FaceNet model
def get_embedding(face):
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    sample = np.expand_dims(face, axis=0)
    input_tensor = tf.convert_to_tensor(sample)
    embedding = facenet_model.signatures["serving_default"](input_tensor)
    embedding = embedding['Bottleneck_BatchNorm'].numpy()
    return embedding[0]

# Function to draw bounding box and label
def draw_box(image, box, label):
    x, y, w, h = box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


def mark_attendance(label_to_store):
    
    # Read the corresponding CSV file
    student_data = pd.read_csv("crnAndName.csv")

    if label_to_store in student_data['CRN'].values:
        # Increment the attendance count
        attendance_info[label_to_store] = attendance_info.get(label_to_store, 0) + 1
        # Increment the existing attendance count in the DataFrame
        student_attendance = student_data.loc[student_data['CRN'] == label_to_store, 'Attendance']
        if pd.isnull(student_attendance).any():
            # If attendance is NaN (i.e., the student hasn't attended before), initialize it to 1
            student_data.loc[student_data['CRN'] == label_to_store, 'Attendance'] = 1
        else:
            # Otherwise, increment the existing attendance count by 1
            student_data.loc[student_data['CRN'] == label_to_store, 'Attendance'] += 1
        # Append the current date to the dates attended
        # current_date = str(date.today())
        current_datetime = datetime.now().strftime('%Y-%m-%d(%H:%M:%S)')
        if 'Dates Attended' in student_data.columns:
            # Check if the student has attended before
            if pd.isnull(student_data.loc[student_data['CRN'] == label_to_store, 'Dates Attended']).any():
                student_data.loc[student_data['CRN'] == label_to_store, 'Dates Attended'] = current_datetime
            else:
                student_data.loc[student_data['CRN'] == label_to_store, 'Dates Attended'] += ', ' + current_datetime
        else:
            student_data['Dates Attended'] = ''
            student_data.loc[student_data['CRN'] == label_to_store, 'Dates Attended'] = current_datetime
        
        # Save the updated attendance information back to the CSV file
        student_data.to_csv("crnAndName.csv", index=False)

    return attendance_info

# Function to perform real-time face classification
def classify_video():
    # Real-time face detection and classification
    cap = cv2.VideoCapture(0)

    label_to_store = None  # Initialize variable to store the label

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces in the frame
        faces = detector.detect_faces(frame)
        
        for face in faces:
            if len(faces) > 0:
                face = faces[0] 
                x, y, w, h = face['box']
                # Preprocess the face for classification
                preprocessed_face = preprocess_image(frame[y:y+h, x:x+w])
                # Classify the face
                label = classify_face(preprocessed_face)
                if label is not None:
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
                    mark_attendance(label_to_store)
                    # Here you can store the label_to_store in your desired way (e.g., in a list)
                    label_to_store = None
                    break
                else:
                    # Get the label of the last classified face and store it
                    if len(faces) > 0:
                        x, y, w, h = faces[0]['box']
                        preprocessed_face = preprocess_image(frame[y:y+h, x:x+w])
                        label = classify_face(preprocessed_face)
                        if label is not None:
                            label_to_store = label

    cap.release()
    cv2.destroyAllWindows()

def find_recent_image(folder_path):
    jpg_files = []
    png_files = []

    # Iterate over all files in the directory
    for file in os.listdir(folder_path):
        if file.lower().endswith('.jpg'):
            jpg_files.append(file)
        elif file.lower().endswith('.png'):
            png_files.append(file)

    # Get the creation time of each file and store in a dictionary
    creation_times = {}
    for file in jpg_files + png_files:
        file_path = os.path.join(folder_path, file)
        creation_time = os.path.getctime(file_path)
        creation_times[file] = creation_time

    # Find the most recent file
    most_recent_file = max(creation_times, key=creation_times.get)
    return most_recent_file

def from_image():
    image_name = find_recent_image('data')
    input_image = cv2.imread('data/'+image_name)


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
            mark_attendance(label)


    cv2.imshow('Face Recognition Result', input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# def find_images_in_folder(folder_path):
#     image_extensions = ['.jpg', '.jpeg', '.png']  # Add more extensions if needed

#     # Iterate over all files in the folder
#     for file_name in os.listdir(folder_path):
#         # Check if the file has one of the image extensions
#         if any(file_name.lower().endswith(ext) for ext in image_extensions):
#             # If yes, add the file path to the list of image files
#             return os.path.join(folder_path, file_name)

# def from_image():
#     # Load known embeddings and labels
#     known_data = np.load("train_data_with_facenet.npz")
#     known_embeddings = known_data['embeddings']
#     known_labels = known_data['labels']

#     image_path = find_images_in_folder('data')
#     # Load the input image
#     input_image = cv2.imread(image_path)

#     # Detect faces in the input image
#     faces = detector.detect_faces(input_image)

#     recognized_names = []  # To store recognized names

#     for face in faces:
#         x, y, w, h = face['box']
#         # Preprocess the face for classification
#         preprocessed_face = preprocess_image(input_image[y:y+h, x:x+w])
#         # Classify the face
#         label_index = classify_face(preprocessed_face, known_embeddings)
#         if label_index is not None:
#             # Get the label corresponding to the index
#             label = known_labels[label_index]
#             # Prompt user to input the name
#             # name = input(f"Enter name for recognized face {label}: ")
#             # recognized_names.append((name, (x, y, w, h)))
#             print(label)
#             # Draw bounding box and label
#             draw_box(input_image, (x, y, w, h),label)

#     # Show the result
#     cv2.imshow('Face Recognition Result', input_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=16)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=4, max=16)])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('This username is already taken.')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=16)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=4, max=16)])
    submit = SubmitField('Login')



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        user = User.query.filter_by(username=username).first()
        if user:
            if user.password == password:
                login_user(user)
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password. Please try again.', 'error')
        else:
            flash('Invalid username or password. Please try again.', 'error')
    return render_template('login.html', form=form)


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, password=form.password.data)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html', form=form)



# Define a list of subjects with their subject code and names


@app.route('/')
def index():
    return render_template('index.html', subjects=subjects)

@app.route('/attendance/<subject_code>', methods=['GET'])
def subject_page(subject_code):
    # Here you can render a page with options to take attendance or view attendance report
    return render_template('subject.html', subject_code=subject_code)

@app.route('/attendance/<subject_code>/take', methods=['GET'])


@app.route('/attendance/<subject_code>/report', methods=['GET'])
def view_report(subject_code):
    # Logic to view attendance report for the specified subject
    return "View attendance report for subject {}".format(subject_code)

@app.route('/take_attendance', methods=['GET'])
def take_attendance():
    return render_template('take_attendance.html')

from flask import Flask, render_template, url_for, redirect, flash, request, jsonify
import os

app.config['UPLOAD_FOLDER'] = 'data'  # Folder where uploaded images will be stored

# Function to save the uploaded image file to the server
def save_uploaded_image(image_data):
    try:
        # Decode the base64-encoded image data
        image_data = image_data.split(',')[1]  # Remove the "data:image/png;base64," prefix
        image_bytes = image_data.encode('utf-8')
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.png'), 'wb') as f:
            f.write(image_bytes)
        return True
    except Exception as e:
        print('Error saving uploaded image:', str(e))
        return False

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Get the image data from the request JSON
        image_data = request.json.get('image_data')
        if image_data:
            # Save the uploaded image to the server
            if save_uploaded_image(image_data):
                # Image was successfully saved
                return jsonify({'success': True, 'message': 'Image uploaded successfully.'}), 200
            else:
                # Error occurred while saving the image
                return jsonify({'success': False, 'message': 'Error uploading image.'}), 500
        else:
            # Image data not found in the request
            return jsonify({'success': False, 'message': 'No image data found in the request.'}), 400
    except Exception as e:
        # Error occurred while processing the request
        print('Error processing image:', str(e))
        return jsonify({'success': False, 'message': 'Error processing image.'}), 500
    
import base64
import uuid  # Import UUID module for generating unique filenames

# Function to save the uploaded image file to the server
def save_uploaded_image(image_data):
    try:
        # Decode the base64-encoded image data
        image_data = image_data.split(',')[1]  # Remove the "data:image/png;base64," prefix
        image_bytes = base64.b64decode(image_data)
        
        # Generate a unique filename for the image
        image_filename = str(uuid.uuid4()) + '.png'

        # Save the image data to a file with the unique filename
        with open(os.path.join(app.config['UPLOAD_FOLDER'], image_filename), 'wb') as f:
            f.write(image_bytes)
        
        return True, image_filename  # Return success status and the filename
    except Exception as e:
        print('Error saving uploaded image:', str(e))
        return False, None




@app.route('/classify')
def classify():
    classify_video()
    return 'Face classification started!'

@app.route('/image')
def image_classify():
    from_image()
    return 'Face classification started!'

if __name__ == '__main__':
    app.run(debug=True)

