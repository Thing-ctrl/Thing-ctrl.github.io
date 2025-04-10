from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
app.secret_key = "your_secret_key" 

MODEL_PATH = "chest_xray_pneumonia_model_ver3.h5" # Đường dẫn đến model đã được huấn luyện
model = load_model(MODEL_PATH)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Kiểm tra định dạng file hợp lệ."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """
    Đọc và tiền xử lý ảnh:
    - Resize ảnh về kích thước (150, 150)
    - Chuẩn hóa giá trị pixel
    - Chuyển đổi ảnh thành mảng numpy với kích thước phù hợp cho model
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, (150, 150))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Không tìm thấy file upload.')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Chưa chọn file nào.')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            upload_folder = "uploads"
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)
            
            processed_image = preprocess_image(filepath)
            if processed_image is None:
                flash("Không thể xử lý ảnh được upload.")
                return redirect(request.url)
            
            # Dự đoán: nếu giá trị > 0.5 thì dự đoán là PNEUMONIA, ngược lại NORMAL
            prediction = model.predict(processed_image)
            # Lấy giá trị dự đoán (scalar) từ mảng kết quả
            pred_value = prediction[0][0]
            if pred_value > 0.5:
                result = "PNEUMONIA"
                confidence = round(pred_value * 100, 2)  # Ví dụ: 82.35%
            else:
                result = "NORMAL"
                confidence = round((1 - pred_value) * 100, 2)
            
            return render_template('result.html', prediction=result, confidence=confidence, filename=filename)
        else:
            flash('File không hợp lệ. Chỉ cho phép ảnh (png, jpg, jpeg, gif).')
            return redirect(request.url)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Hiển thị file đã upload."""
    return send_from_directory("uploads", filename)

if __name__ == '__main__':
    app.run(debug=True)
