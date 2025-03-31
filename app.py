from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
import os
import base64
from tensorflow.keras.layers import Multiply
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(1,), initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(score, axis=1)
        return Multiply()([inputs, attention_weights])

    def get_config(self):
        return super(Attention, self).get_config()


MODEL_PATH = "atp-aceguard.keras"
model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"Attention": Attention})
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()


def extract_pose_keypoints(img):
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        # if results.pose_landmarks:
        #     return np.array([val for lm in results.pose_landmarks.landmark for val in (lm.x, lm.y, lm.z)])
        if results.pose_landmarks:
            keypoints = np.array([val for lm in results.pose_landmarks.landmark for val in (lm.x, lm.y, lm.z)])

            # Normalize keypoints using Min-Max Scaling
            keypoints = (keypoints - np.min(keypoints)) / (np.max(keypoints) - np.min(keypoints) + 1e-6)

            print(f"✅ Normalized Keypoints: {keypoints[:10]}")  # Debug
            return keypoints
    except Exception as e:
        print(f"Error extracting keypoints: {e}")
    return None


@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        keypoints = extract_pose_keypoints(img)
        if keypoints is None:
            return jsonify({"error": "No keypoints detected in image"}), 400

        keypoints = keypoints.reshape(1, -1)
        prediction = model.predict(keypoints)[0][0]
        confidence = round(float(prediction * 100), 2)
        threshold = 0.55
        result_label = "Proper" if prediction > threshold else "Improper"
        print(prediction)
        print(confidence)
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        _, buffer = cv2.imencode(".jpg", img)
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "label": result_label,
            "confidence": confidence,
            "skeleton_image": f"data:image/jpeg;base64,{encoded_image}"
        })
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)
