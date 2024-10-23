'''
对输入的图片进行人脸识别
'''

import cv2
import numpy as np
import tensorflow as tf

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def detect_faces(image, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    return faces

def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (64, 64))
    face = face.reshape(1, 64, 64, 1)
    face = face / 255.0  # Normalize the image
    return face

def draw_faces(image, faces, model):
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        preprocessed_face = preprocess_face(face)
        prediction = model.predict(preprocessed_face)
        label = np.argmax(prediction)
        color = (0, 255, 0) if label == 1 else (0, 0, 255)  # Green for label 1, Red for label 0
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

def main(image_path):
    # Load Haar Cascade Classifier
    face_cascade = cv2.CascadeClassifier(
        r'D:\Program Files\Python38\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')

    # Load the trained model
    model = load_model('cnn_model.h5')

    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Unable to load image")
        return

    # Detect faces in the image
    faces = detect_faces(image, face_cascade)

    # Draw faces with different colors based on the label
    draw_faces(image, faces, model)

    # Display the result
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = r"D:\Desktop\mmexport1720788544657.jpg"  # Replace with your image path
    main(image_path)