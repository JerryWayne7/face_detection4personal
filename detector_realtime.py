import cv2
import numpy as np
import tensorflow as tf

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    return faces

def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (64, 64))
    face = face.reshape(1, 64, 64, 1)
    face = face / 255.0  # Normalize the image
    return face

def draw_faces(frame, faces, model):
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        preprocessed_face = preprocess_face(face)
        prediction = model.predict(preprocessed_face)
        label = np.argmax(prediction)
        color = (0, 255, 0) if label == 1 else (0, 0, 255)  # Green for label 1, Red for label 0
        if label:
            print("\a")
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

def main():
    # Load Haar Cascade Classifier
    face_cascade = cv2.CascadeClassifier(
        r'D:\Program Files\Python38\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')

    # Load the trained model
    model = load_model('cnn_model.h5')

    # Open the camera
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video from camera")
            break

        faces = detect_faces(frame, face_cascade)

        # Draw faces with different colors based on the label
        draw_faces(frame, faces, model)

        cv2.imshow('Real-time Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()