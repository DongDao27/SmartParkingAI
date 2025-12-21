import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import datetime
import json
import os
import re
import time
from flask import Flask, render_template, Response, jsonify, request

# --- KHỞI TẠO FLASK ---
app = Flask(__name__)

# --- CẤU HÌNH ---
DB_FILE = "parking_data.json"
IMG_FOLDER = "static/parked_images" 
MODEL_PATH = "best.pt" # Đảm bảo file này nằm cùng thư mục

if not os.path.exists(IMG_FOLDER): os.makedirs(IMG_FOLDER)

# Mã tỉnh Việt Nam (Dùng để kiểm tra biển số có thật không)
PROVINCE_CODE = [
    "11", "12", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", 
    "30", "31", "32", "33", "34", "35", "36", "37", "38", "40", "41", "43", "47", "48", "49", "50", "51", "52", 
    "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", 
    "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "88", "89", 
    "90", "92", "93", "94", "95", "97", "98", "99"
]

# --- CLASS XỬ LÝ ẢNH (ĐÃ ĐƠN GIẢN HÓA & TỐI ƯU) ---
class ImageProcessor:
    @staticmethod
    def preprocess_for_ocr(img):
        """
        Quy trình xử lý chuẩn cho EasyOCR:
        1. Phóng to ảnh (Quan trọng nhất)
        2. Chuyển xám
        3. Tăng tương phản
        """
        # 1. Phóng to ảnh (Upscale) để OCR nhìn rõ hơn
        height, width = img.shape[:2]
        if width < 300: # Nếu ảnh quá bé
            scale = 3.0 # Phóng to gấp 3 lần
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # 2. Chuyển sang ảnh xám (Grayscale)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 3. Giảm nhiễu nhẹ (Bilateral Filter giữ cạnh chữ tốt hơn Gaussian)
        denoised = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # 4. Tăng độ tương phản (Histogram Equalization) nhưng không làm cháy ảnh
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(denoised)
        
        # LƯU Ý: KHÔNG DÙNG THRESHOLD (Đen trắng) VÌ SẼ LÀM MẤT NÉT CHỮ
        return contrast

# --- CLASS AI ENGINE ---
class LicensePlateRecognizer:
    def __init__(self, model_path=MODEL_PATH):
        print(">> Đang tải Model YOLO...")
        self.model = YOLO(model_path)
        print(">> Đang tải EasyOCR...")
        # Dùng model tiếng Anh (en) là đủ cho số và chữ cái
        self.reader = easyocr.Reader(['en'], gpu=True)

    def strict_correction(self, text):
        """Logic sửa lỗi cực kỳ chặt chẽ"""
        # 1. Chỉ giữ ký tự Chữ và Số
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        if len(text) < 5: return None # Quá ngắn -> Rác

        chars = list(text)
        
        # Bảng thay thế ký tự hay nhầm
        # SỐ -> CHỮ (cho vị trí Series)
        num_to_char = {'0':'D', '1':'I', '2':'Z', '4':'A', '5':'S', '8':'B'}
        # CHỮ -> SỐ (cho vị trí Mã tỉnh và Số cuối)
        char_to_num = {'D':'0', 'O':'0', 'Q':'0', 'I':'1', 'Z':'2', 'B':'8', 'A':'4', 'S':'5', 'G':'6', 'L':'1'}

        # --- LOGIC VỊ TRÍ (QUAN TRỌNG) ---
        # Vị trí 0,1: Mã tỉnh -> BẮT BUỘC LÀ SỐ
        for i in range(min(2, len(chars))):
            if chars[i] in char_to_num: chars[i] = char_to_num[chars[i]]

        # Vị trí 2: Series -> THƯỜNG LÀ CHỮ (Trừ biển LD, KT...)
        if len(chars) > 2 and chars[2] in num_to_char:
            chars[2] = num_to_char[chars[2]]

        # Vị trí 3 trở đi: Số xe -> BẮT BUỘC LÀ SỐ
        for i in range(3, len(chars)):
            if chars[i] in char_to_num: chars[i] = char_to_num[chars[i]]
        
        final_plate = "".join(chars)

        # KIỂM TRA MÃ TỈNH CÓ HỢP LỆ KHÔNG
        if final_plate[:2] not in PROVINCE_CODE:
            return None # Mã tỉnh sai -> Rác

        return final_plate

    def detect(self, frame):
        # Chạy YOLO
        results = self.model(frame, conf=0.25, verbose=False)[0] # Giảm conf để bắt nhạy hơn
        best_plate = None
        annotated_frame = frame.copy()
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Bỏ qua box quá nhỏ (nhiễu)
            if (x2 - x1) < 30 or (y2 - y1) < 15: continue

            # Cắt ảnh biển số (Mở rộng vùng cắt 5px để không bị mất viền chữ)
            h_img, w_img = frame.shape[:2]
            y1_crop = max(0, y1 - 5)
            y2_crop = min(h_img, y2 + 5)
            x1_crop = max(0, x1 - 5)
            x2_crop = min(w_img, x2 + 5)
            
            plate_img = frame[y1_crop:y2_crop, x1_crop:x2_crop]
            
            # Xử lý ảnh
            processed = ImageProcessor.preprocess_for_ocr(plate_img)

            # Phân loại biển Vuông / Dài để đọc
            ratio = plate_img.shape[1] / plate_img.shape[0]
            try:
                raw_text = ""
                if ratio < 2.5: 
                    # --- BIỂN VUÔNG (2 dòng) ---
                    # Cắt đôi theo chiều dọc
                    half = processed.shape[0] // 2
                    top_img = processed[:half, :]
                    bot_img = processed[half:, :]
                    
                    # Dòng trên: Chữ + Số
                    res_top = self.reader.readtext(top_img, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                    # Dòng dưới: CHỈ SỐ (Fix cứng để tránh đọc sai)
                    res_bot = self.reader.readtext(bot_img, detail=0, allowlist='0123456789.')
                    
                    raw_text = "".join(res_top) + "".join(res_bot)
                else:
                    # --- BIỂN DÀI (1 dòng) ---
                    res = self.reader.readtext(processed, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.')
                    raw_text = "".join(res)
                
                # Sửa lỗi
                final_plate = self.strict_correction(raw_text)
                
                if final_plate:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    best_plate = final_plate
                    # Tìm thấy biển hợp lệ thì dừng, không cần quét box khác
                    break 

            except Exception as e:
                print(f"Lỗi OCR: {e}")
                continue
            
        return best_plate, annotated_frame

# --- CLASS DATABASE (GIỮ NGUYÊN) ---
class ParkingDatabase:
    def __init__(self):
        self.load()
    def load(self):
        if os.path.exists(DB_FILE):
            with open(DB_FILE, 'r', encoding='utf-8') as f: self.data = json.load(f)
        else: self.data = {"active": {}, "history": [], "statistics": {"total_revenue": 0}}
    def save(self):
        with open(DB_FILE, 'w', encoding='utf-8') as f: json.dump(self.data, f, indent=4, ensure_ascii=False)
    
    def check_in(self, plate, img_path):
        if plate in self.data["active"]: return False, "Xe đang ở trong bãi!"
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.data["active"][plate] = {"in_time": now, "img": img_path}
        self.save()
        return True, f"Giờ vào: {now}"

    def check_out(self, plate):
        if plate not in self.data["active"]: return False, "Không tìm thấy xe!"
        info = self.data["active"].pop(plate)
        t_in = datetime.datetime.strptime(info["in_time"], "%Y-%m-%d %H:%M:%S")
        minutes = int((datetime.datetime.now() - t_in).total_seconds() / 60)
        cost = max(5000, minutes * 1000) 
        self.data["history"].append({"plate": plate, "in": info["in_time"], "cost": cost})
        self.data["statistics"]["total_revenue"] += cost
        self.save()
        return True, f"Phí: {cost:,} đ ({minutes}p)"

# --- KHỞI TẠO ---
ai_engine = LicensePlateRecognizer()
db = ParkingDatabase()
camera = cv2.VideoCapture(0) 

global_frame = None
is_camera_paused = False 

def generate_frames():
    global global_frame, is_camera_paused
    while True:
        if not is_camera_paused:
            success, frame = camera.read()
            if success:
                global_frame = frame
        
        if global_frame is not None:
            try:
                ret, buffer = cv2.imencode('.jpg', global_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except: pass
        time.sleep(0.04)

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    global global_frame, is_camera_paused
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is not None:
        global_frame = img   
        is_camera_paused = True 
        plate, _ = ai_engine.detect(img.copy())
        return jsonify({"status": "success", "plate": plate if plate else "Không đọc được"})
    return jsonify({"status": "error", "msg": "File lỗi"})

@app.route('/api/reset_camera', methods=['POST'])
def reset_camera():
    global is_camera_paused
    is_camera_paused = False
    return jsonify({"status": "success"})

@app.route('/api/action', methods=['POST'])
def action():
    action_type = request.json.get('type')
    if global_frame is None: return jsonify({"status": "error", "msg": "Không có ảnh"})
    
    # Nhận diện
    plate, ann_img = ai_engine.detect(global_frame.copy())
    
    if not plate:
        return jsonify({"status": "error", "msg": "Không tìm thấy biển số hoặc biển số quá mờ."})

    # Format hiển thị đẹp
    fmt_plate = f"{plate[:3]}-{plate[3:6]}.{plate[6:]}" if len(plate) >= 7 else plate
    
    msg = ""
    status = "success"
    if action_type == 'in':
        img_name = f"{plate}_{int(time.time())}.jpg"
        img_path = os.path.join(IMG_FOLDER, img_name)
        cv2.imwrite(img_path, global_frame) 
        ok, res = db.check_in(fmt_plate, img_path)
        if not ok: status = "error"
        msg = res
    else:
        ok, res = db.check_out(fmt_plate)
        if not ok: status = "error"
        msg = res
        
    stats = {"active": len(db.data["active"]), "revenue": db.data["statistics"]["total_revenue"]}
    return jsonify({"status": status, "plate": fmt_plate, "msg": msg, "stats": stats})

@app.route('/api/data')
def get_data():
    return jsonify(db.data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)