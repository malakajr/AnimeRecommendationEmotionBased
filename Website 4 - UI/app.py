from flask import Flask, render_template, request, jsonify
import base64
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

app = Flask(__name__, static_url_path='/static')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    json_data = request.json
    photo_data = json_data['photo']  # access the 'photo' key

    with open('uploaded_photo.png', 'wb') as f:
        f.write(base64.b64decode(photo_data.split(',')[1]))

    face_classifier = cv2.CascadeClassifier(
        'haarcascades/haarcascade_frontalface_default.xml')
    emotion_model = load_model('emotion_detection_model_50epochs.h5')
    gender_model = load_model('gender_model_50epochs.h5')

    class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
    gender_labels = ['Male', 'Female']

    img = cv2.imread("uploaded_photo.png")
    if img is None:
        return jsonify({'error': 'Image not found'})

    labels = []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Get image ready for prediction
        roi = roi_gray.astype('float') / 255.0  # Scale
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        label = ""
        # Yields one hot encoded result for 7 classes
        preds = emotion_model.predict(roi)[0]
        label = class_labels[preds.argmax()]  # Find the label
        label_position = (x, y)
        cv2.putText(img, label, label_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Gender
        gender_label = ""
        roi_color = img[y:y + h, x:x + w]
        roi_color = cv2.resize(roi_color, (200, 200),
                               interpolation=cv2.INTER_AREA)
        gender_predict = gender_model.predict(
            np.array(roi_color).reshape(-1, 200, 200, 3))
        gender_predict = (gender_predict >= 0.5).astype(int)[:, 0]
        gender_label = gender_labels[gender_predict[0]]
        # 50 pixels below to move the label outside the face
        gender_label_position = (x, y + h + 50)
        cv2.putText(img, gender_label, gender_label_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return jsonify({'message': 'Image shown', 'gender': gender_label, 'emotion': label})


@app.route('/result', methods=['POST'])
def result():
    face_classifier = cv2.CascadeClassifier(
        'haarcascades/haarcascade_frontalface_default.xml')
    emotion_model = load_model('emotion_detection_model_50epochs.h5')
    gender_model = load_model('gender_model_50epochs.h5')

    class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
    gender_labels = ['Male', 'Female']

    cap = cv2.imread("uploaded_photo.png")
    if cap is None:
        return jsonify({'error': 'Image not found'})

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        labels = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48),
                                  interpolation=cv2.INTER_AREA)

            # Get image ready for prediction
            roi = roi_gray.astype('float') / 255.0  # Scale
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Yields one hot encoded result for 7 classes
            preds = emotion_model.predict(roi)[0]
            label = class_labels[preds.argmax()]  # Find the label
            label_position = (x, y)
            cv2.putText(frame, label, label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Gender
            gender_label = ""
            roi_color = frame[y:y + h, x:x + w]
            roi_color = cv2.resize(roi_color, (200, 200),
                                   interpolation=cv2.INTER_AREA)
            gender_predict = gender_model.predict(
                np.array(roi_color).reshape(-1, 200, 200, 3))
            gender_predict = (gender_predict >= 0.5).astype(int)[:, 0]
            gender_label = gender_labels[gender_predict[0]]
            # 50 pixels below to move the label outside the face
            gender_label_position = (x, y + h + 50)
            cv2.putText(frame, gender_label, gender_label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Emotion Detector', cap)
        cv2.waitKey(0)

    return jsonify({'message': 'Image shown', 'gender': gender_label})


@app.route('/resultpage')
def resultpage():
    gender = request.args.get('gender')
    emotion = request.args.get('emotion')
    return render_template('result.html', gender=gender, emotion=emotion)


if __name__ == '__main__':
    app.run()
