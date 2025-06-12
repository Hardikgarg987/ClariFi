# app.py

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

from enhance import enhance_audio

UPLOAD_FOLDER = 'uploads/'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/enhance", methods=['POST'])
def enhance():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(path)

    output_path = f"outputs/enhanced_{filename}"
    enhance_audio(path, output_path)

    return jsonify({"message": "Audio enhanced", "output": output_path})

if __name__ == "__main__":
    app.run(debug=True)
