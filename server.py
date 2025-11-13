from flask import Flask, request, jsonify
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import io, base64

app = Flask(__name__)

# Load model .tflite
interpreter = tflite.Interpreter(model_path="mobilenetv5.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        img_data = data["image"]

        # Cắt prefix nếu có
        if "," in img_data:
            img_data = img_data.split(",")[1]

        # Decode base64
        try:
            img_bytes = base64.b64decode(img_data)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            print("🔥 Base64 decode error:", e)
            return jsonify({"error": "invalid_image"}), 400

        # Resize
        image = image.resize((224, 224))
        image = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)

        # TFLite inference
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        result = float(output[0][0])

        # Output label
        label = "Thiếu máu" if result > 0.5 else "Bình thường"

        return jsonify({'result': label, 'confidence': round(result, 3)})

    except Exception as e:
        print("🔥 SERVER ERROR:", e)
        return jsonify({"error": "server_error"}), 500

