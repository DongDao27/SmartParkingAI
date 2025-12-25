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
MODEL_PATH = "best.pt"

if not os.path.exists(IMG_FOLDER): os.makedirs(IMG_FOLDER)

# Danh sách mã tỉnh
PROVINCE_CODE = [
    "11", "12", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", 
    "30", "31", "32", "33", "34", "35", "36", "37", "38", "40", "41", "43", "47", "48", "49", "50", "51", "52", 
    "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", 
    "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "88", "89", 
    "90", "92", "93", "94", "95", "97", "98", "99"
]

# --- XỬ LÝ ẢNH ---
class ImageProcessor:
    @staticmethod
    def preprocess_for_ocr(img):
        h, w = img.shape[:2]
        if w < 600:
            scale = 600 / w
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharp = cv2.filter2D(enhanced, -1, kernel)
        
        return sharp 

# --- AI ENGINE ---
class LicensePlateRecognizer:
    def __init__(self, model_path=MODEL_PATH):
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader(['en'], gpu=True)

    def strict_correction(self, text):
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        if len(text) < 5: return None
        chars = list(text)
        
        char_to_num = {'O':'0', 'D':'0', 'Q':'0', 'I':'1', 'Z':'2', 'B':'8', 'A':'4', 'S':'5', 'G':'6', 'L':'1'}
        num_to_char = {'0':'D', '1':'I', '2':'Z', '4':'A', '5':'S', '8':'B'}
        
        for i in range(min(2, len(chars))):
            if chars[i] in char_to_num: chars[i] = char_to_num[chars[i]]
            
        if len(chars) > 2 and chars[2] in num_to_char: 
            chars[2] = num_to_char[chars[2]]

        for i in range(4, len(chars)):
            if chars[i] in char_to_num: chars[i] = char_to_num[chars[i]]

        final_text = "".join(chars)
        
        if final_text[:2] not in PROVINCE_CODE:
            return None 

        return final_text

    def detect(self, frame):
        results = self.model(frame, conf=0.35, verbose=False)[0]
        best_plate = None
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            processed = ImageProcessor.preprocess_for_ocr(crop)
            try:
                res = self.reader.readtext(processed, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.')
                raw = "".join(res)
                final = self.strict_correction(raw)
                
                if final:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    best_plate = final
                    break 
            except: continue
        return best_plate, frame

# --- DATABASE ---
class ParkingDatabase:
    def __init__(self):
        self.load()
    def load(self):
        if os.path.exists(DB_FILE):
            try:
                with open(DB_FILE, 'r', encoding='utf-8') as f: self.data = json.load(f)
            except: self.data = {"active": {}, "history": [], "stats": {"revenue": 0}}
        else: self.data = {"active": {}, "history": [], "stats": {"revenue": 0}}
    
    def save(self):
        with open(DB_FILE, 'w', encoding='utf-8') as f: json.dump(self.data, f, indent=4, ensure_ascii=False)
    
    def check_in(self, plate, img_path):
        if plate in self.data["active"]: return False, "Xe này đã có trong bãi!", 0, 0
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.data["active"][plate] = {"in_time": now, "img": img_path}
        self.save()
        return True, f"Mời vào ({now})", 0, 0

    def check_out(self, plate):
        if plate not in self.data["active"]: return False, "Xe chưa Check-in!", 0, 0
        info = self.data["active"].pop(plate)
        t_in = datetime.datetime.strptime(info["in_time"], "%Y-%m-%d %H:%M:%S")
        
        # --- LOGIC TÍNH TIỀN MỚI ---
        duration = datetime.datetime.now() - t_in
        total_seconds = duration.total_seconds()
        hours = total_seconds / 3600
        minutes = int(total_seconds / 60)
        
        if hours < 10:
            cost = 5000
        elif hours < 24:
            cost = 10000
        else:
            days = int(np.ceil(hours / 24))
            cost = days * 30000
        # ---------------------------
        
        self.data["history"].append({
            "plate": plate, "in": info["in_time"], 
            "out": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "cost": cost
        })
        self.data["stats"]["revenue"] += cost
        self.save()
        return True, "Mời xe ra", minutes, cost

# --- SETUP ---
ai_engine = LicensePlateRecognizer()
db = ParkingDatabase()
camera = cv2.VideoCapture(0) 

global_frame = None
is_paused = False

def gen_frames():
    global global_frame, is_paused
    while True:
        if not is_paused:
            success, frame = camera.read()
            if success: global_frame = frame
        if global_frame is not None:
            try:
                ret, buffer = cv2.imencode('.jpg', global_frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except: pass
        time.sleep(0.04)

# --- ROUTES ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    global global_frame, is_paused
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is not None:
        global_frame = img; is_paused = True
        plate, _ = ai_engine.detect(img.copy())
        return jsonify({"status": "success", "plate": plate if plate else "---"})
    return jsonify({"status": "error", "msg": "Lỗi file"})

@app.route('/api/reset_camera', methods=['POST'])
def reset_camera():
    global is_paused; is_paused = False
    return jsonify({"status": "success"})

@app.route('/api/action', methods=['POST'])
def action():
    act_type = request.json.get('type')
    if global_frame is None: return jsonify({"status": "error", "msg": "Mất tín hiệu"})
    
    plate, ann_img = ai_engine.detect(global_frame.copy())
    if not plate: return jsonify({"status": "error", "msg": "Không đọc được biển số"})
    
    if len(plate) >= 8 and plate[3].isdigit():
        fmt_plate = f"{plate[:4]}-{plate[4:7]}.{plate[7:]}"
    elif len(plate) >= 7:
        fmt_plate = f"{plate[:3]}-{plate[3:6]}.{plate[6:]}"
    else:
        fmt_plate = plate

    if act_type == 'in':
        path = f"{IMG_FOLDER}/{plate}_{int(time.time())}.jpg"
        cv2.imwrite(path, global_frame)
        ok, msg, mins, cost = db.check_in(fmt_plate, path)
    else:
        ok, msg, mins, cost = db.check_out(fmt_plate)
    
    return jsonify({
        "status": "success" if ok else "error",
        "plate": fmt_plate, "msg": msg, "minutes": mins, "cost": cost,
        "stats": {"active": len(db.data["active"]), "revenue": db.data["stats"]["revenue"]}
    })

@app.route('/api/data')
def get_data(): return jsonify(db.data)

if __name__ == '__main__':
    app.run(port=5000, debug=True)