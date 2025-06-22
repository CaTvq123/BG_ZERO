from flask import Flask, request, send_file, render_template
from PIL import Image
import numpy as np
import io
import cv2
from rembg import remove, new_session
import requests
from dotenv import load_dotenv
import os

load_dotenv()

# 🚀 Lấy API key từ biến môi trường
CLOUDMERSIVE_KEY = os.getenv("CLOUDMERSIVE_API_KEY")

# 🚀 Tạo Flask app
app = Flask(__name__)

# 🚀 Tạo session với mô hình mạnh hơn U2Net
session = new_session("isnet-general-use")  # Xịn hơn u2net, tách được cả người + vật

# 🔍 Tự động phân loại ảnh bằng API

def detect_image_type(img_bytes):
    url = "https://api.cloudmersive.com/image/recognize/describe"
    headers = {"Apikey": CLOUDMERSIVE_KEY}
    files = {'imageFile': ('image.png', img_bytes)}

    try:
        res = requests.post(url, headers=headers, files=files)
        desc = res.json().get("BestOutcome", {}).get("Description", "").lower()

        if "person" in desc or "face" in desc:
            return "real"
        elif "map" in desc or "icon" in desc or "cartoon" in desc:
            return "icon"
        else:
            return "real"
    except Exception as e:
        print("Lỗi phân loại ảnh:", e)
        return "real"

# 🧠 Hàm xử lý tách nền thông minh

def smart_background_removal(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    except:
        raise ValueError("Không đọc được ảnh. Có thể ảnh bị hỏng hoặc sai định dạng.")

    if img.width < 100 or img.height < 100:
        raise ValueError("Ảnh quá nhỏ. Vui lòng chọn ảnh lớn hơn 100x100.")

    category = detect_image_type(img_bytes)

    if category == "icon":
        img_np = np.array(img)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(hsv, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.medianBlur(mask, 3)
        result = np.zeros_like(img_np)
        for c in range(3):
            result[:, :, c] = np.where(mask > 0, img_np[:, :, c], 0)
        result[:, :, 3] = mask
        final = Image.fromarray(result)
    else:
        try:
            result_bytes = remove(img_bytes, session=session)
            result = Image.open(io.BytesIO(result_bytes)).convert("RGBA")

            result_np = np.array(result)
            alpha = result_np[:, :, 3]
            blurred = cv2.GaussianBlur(alpha, (5, 5), 0)
            _, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

            original_np = np.array(img)
            final_np = np.zeros_like(original_np)
            for c in range(3):
                final_np[:, :, c] = np.where(binary > 0, original_np[:, :, c], 0)
            final_np[:, :, 3] = binary
            final = Image.fromarray(final_np)
        except Exception as e:
            print("Lỗi xử lý rembg:", e)
            raise ValueError("Không thể xử lý ảnh nền. Vui lòng thử ảnh khác.")

    buf = io.BytesIO()
    final.save(buf, format="PNG")
    buf.seek(0)
    return buf

# 🚪 Trang chủ
@app.route('/')
def index():
    return render_template('index.html')

# 📤 API upload ảnh
@app.route('/smart-upload', methods=['POST'])
def smart_upload():
    if 'image' not in request.files:
        return 'No file uploaded', 400

    file = request.files['image']
    if file.filename == '':
        return 'No file selected', 400

    try:
        img_bytes = file.read()
        output = smart_background_removal(img_bytes)
        return send_file(output, mimetype='image/png')
    except ValueError as ve:
        return str(ve), 400
    except Exception as e:
        print("Lỗi tổng quát:", e)
        return 'Internal server error', 500

# ▶️ Chạy app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

# ✅ Force change for Render porta
