from flask import Flask, render_template, request, url_for
from enhance import enhance_audio
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ENHANCED_FOLDER = 'static/enhanced'
ALLOWED_EXTENSIONS = {'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ENHANCED_FOLDER'] = ENHANCED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ENHANCED_FOLDER, exist_ok=True)

# File validation
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home page: Upload Form
@app.route('/')
def index():
    return render_template('index.html')

# Handle Upload and Enhancement
@app.route('/enhance', methods=['POST'])
def enhance():
    if 'audiofile' not in request.files:
        return "No file uploaded", 400
    file = request.files['audiofile']

    if file and allowed_file(file.filename):
        filename = file.filename
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['ENHANCED_FOLDER'], 'enhanced_' + filename)

        # Save file
        file.save(input_path)

        # Enhance and get metrics
        seg_snr, pesq_val, stoi_val, spectrogram_path = enhance_audio(input_path, output_path=output_path)

        return render_template('result.html',
            original_file=url_for('static', filename=f'uploads/{filename}'),
            enhanced_file=url_for('static', filename=f'enhanced/enhanced_{filename}'),
            seg_snr=f"{seg_snr:.2f}",
            pesq_val=f"{pesq_val:.2f}",
            stoi_val=f"{stoi_val:.2f}",
            spectrogram_path=spectrogram_path
        )

    return "Invalid file type", 400

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
