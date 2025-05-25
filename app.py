from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# app = Flask(__name__)

# # Load your model
# model = tf.keras.models.load_model('Model/emotion_cnn_model_with_k_fold_validation.h5')

# # Define emotion labels (update according to your model)
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image part'}), 400

#     file = request.files['image']
#     img = Image.open(file.stream).convert('L')  # Convert to grayscale if your model expects that

#     img = img.resize((48, 48))  # Resize to match model input
#     img_array = np.array(img) / 255.0
#     img_array = img_array.reshape(1, 48, 48, 1)  # Assuming input shape is (48, 48, 1)

#     prediction = model.predict(img_array)
#     predicted_label = emotion_labels[np.argmax(prediction)]

#     return jsonify({'prediction': predicted_label})

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

app = Flask(__name__)

# Load your pretrained model
model = tf.keras.models.load_model('Model/emotion_cnn_model_with_k_fold_validation.h5')

# Define labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/')
def index():
    return "✅ Flask Emotion Prediction API is running."


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file.stream).convert('L')  # Convert to grayscale
    img = img.resize((48, 48))  # Resize to match model input
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape(1, 48, 48, 1)

    prediction = model.predict(img_array)
    predicted_index = int(np.argmax(prediction))
    emotion = emotion_labels[predicted_index]

    return jsonify({
        'emotion': emotion,
        'confidence': float(np.max(prediction))
    })

# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)