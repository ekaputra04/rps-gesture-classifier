from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io

# Inisialisasi Flask app dan load model
app = Flask(__name__)
model = load_model('./model.h5')

# Daftar kelas prediksi
class_names = ["kertas", "batu", "gunting"]

@app.route('/predict', methods=['POST'])
def predict():
    # Cek apakah gambar ada dalam header
    if 'image' not in request.files:
        return jsonify({"error": "No image found in request"}), 400
    
    # Ambil gambar dari request
    image_file = request.files['image'].read()
    image = Image.open(io.BytesIO(image_file)).convert('RGB')
    
    # Preprocess gambar
    image = image.resize((150, 150))  # Ubah sesuai input size model
    image = img_to_array(image) / 255.0  # Normalisasi
    image = np.expand_dims(image, axis=0)  # Tambahkan dimensi batch
    
    # Prediksi
    predictions = model.predict(image)
    accuracy = float(np.max(predictions))  # Akurasi prediksi tertinggi
    predicted_class = class_names[np.argmax(predictions)]  # Prediksi label
    
    # Return hasil dalam format JSON
    return jsonify({
        "accuracy": accuracy,
        "prediction": predicted_class
    })

# Jalankan Flask app
if __name__ == '__main__':
    app.run(debug=True)
