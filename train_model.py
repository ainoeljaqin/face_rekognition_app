import cv2
import os
import numpy as np

# Initialize face recognizer and face detector
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to load dataset and train model
def prepare_training_data(data_folder_path):
    faces = []
    labels = []
    label_names = []
    current_label = 0

    # Iterate over each person's folder
    for person_name in os.listdir(data_folder_path):
        person_path = os.path.join(data_folder_path, person_name)
        
        # Skip if not a directory
        if not os.path.isdir(person_path):
            continue

        # Add person's name to label list
        label_names.append(person_name)
        
        # Iterate over each image in person's folder
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            # Crop and save detected faces
            for (x, y, w, h) in faces_rect:
                faces.append(gray[y:y+h, x:x+w])
                labels.append(current_label)
        current_label += 1
    
    # Train the face recognizer
    face_recognizer.train(faces, np.array(labels))
    
    # Return the list of label names for later use
    return label_names

# Path ke folder dataset
if __name__ == "__main__":
    data_folder_path = "dataset"  # Folder utama dataset
    label_names = prepare_training_data(data_folder_path)
    
    # Save trained model
    face_recognizer.write("trained_model.yml")
    np.save("label_names.npy", label_names)
