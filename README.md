# 🚗 SMART PARKING AI CONTROL CENTER
### Hệ Thống Quản Lý Bãi Đỗ Xe Thông Minh & Tự Động Hóa
**Computer Vision • Artificial Intelligence • IoT Simulation**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Framework-000000?style=flat&logo=flask&logoColor=white)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-00FFFF?style=flat&logo=yolo&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📑 Mục Lục
1. [Giới Thiệu](#-giới-thiệu)
2. [Tính Năng Nổi Bật](#-tính-năng-nổi-bật)
3. [Công Nghệ Sử Dụng](#-công-nghệ-sử-dụng)
4. [Luồng Xử Lý & Thuật Toán](#-luồng-xử-lý--thuật-toán)
5. [Hướng Dẫn Cài Đặt](#-hướng-dẫn-cài-đặt)
6. [Cấu Trúc Dự Án](#-cấu-trúc-dự-án)
7. [Đội Ngũ Phát Triển](#-đội-ngũ-phát-triển)

---

## 📖 Giới Thiệu
**Smart Parking AI** là giải pháp phần mềm quản lý bãi đỗ xe tự động, ứng dụng công nghệ **Thị giác máy tính (Computer Vision)** để thay thế quy trình kiểm soát thẻ từ truyền thống.

Hệ thống giúp tự động hóa toàn bộ quy trình từ lúc xe vào (Check-in) đến lúc xe ra (Check-out), nhận diện biển số chính xác, mô phỏng điều khiển Barie và tính toán chi phí dịch vụ minh bạch theo thời gian thực.

## ✨ Tính Năng Nổi Bật

### 🧠 1. Nhận Diện Biển Số AI (ANPR)
- **Real-time Recognition:** Quét biển số trực tiếp từ Webcam hoặc Camera IP với độ trễ thấp.
- **Hỗ trợ đa nguồn:** Cho phép tải lên ảnh tĩnh (Image Upload) để kiểm thử hoặc xử lý nguội.
- **Xử lý ảnh nâng cao:** Tự động cân bằng sáng, khử nhiễu và làm nét ảnh trước khi đọc.

### 🖥️ 2. Giao Diện Điều Khiển (Dashboard)
- **Thiết kế hiện đại:** Giao diện **Dark Mode** (tối màu) sang trọng, phù hợp với các trung tâm giám sát.
- **Mô phỏng IoT:** Hiển thị trạng thái Barie (Đóng/Mở) và Đèn tín hiệu (Xanh/Đỏ) ngay trên màn hình.
- **Thống kê:** Cập nhật số lượng xe trong bãi và tổng doanh thu theo thời gian thực.

### 💰 3. Quản Lý Tài Chính & Hóa Đơn
- **Tính tiền tự động:** Hệ thống tự động tính thời gian lưu trú (phút/giờ).
- **Hóa đơn điện tử:** Hiển thị chi tiết thời gian và số tiền khách cần trả khi Check-out.
- **Lưu trữ lịch sử:** Toàn bộ lượt xe ra vào đều được lưu lại để tra cứu.

---

## 🛠️ Công Nghệ Sử Dụng

| Lĩnh Vực | Công Nghệ / Thư Viện | Mô Tả |
| :--- | :--- | :--- |
| **Backend** | **Python**, **Flask** | Xử lý logic server, API và điều phối hệ thống. |
| **AI Core** | **YOLOv8** | Object Detection: Xác định vị trí biển số trong khung hình. |
| **OCR** | **EasyOCR**, **PyTorch** | Optical Character Recognition: Đọc ký tự từ ảnh. |
| **Image Proc** | **OpenCV**, **NumPy** | Tiền xử lý ảnh (Grayscale, Resize, Histogram Equalization). |
| **Frontend** | **HTML5/CSS3**, **Bootstrap 5** | Xây dựng giao diện Responsive, Grid System. |
| **Database** | **JSON** (NoSQL) | Lưu trữ dữ liệu dạng file (Nhanh, nhẹ, không cần cài DB Server). |
| **Tools** | **Git**, **VS Code** | Quản lý mã nguồn và môi trường phát triển. |

---

## ⚙️ Luồng Xử Lý & Thuật Toán

### 1. Quy Trình Nhận Diện (Vision Pipeline)
1.  **Input:** Nhận luồng video hoặc ảnh đầu vào.
2.  **Preprocessing:** * Resize ảnh về kích thước chuẩn.
    * Chuyển sang ảnh xám (Grayscale).
    * Tăng độ tương phản (CLAHE) và làm nét (Sharpening).
3.  **Detection:** Model YOLOv8 phát hiện khung hình chữ nhật chứa biển số.
4.  **Recognition:** EasyOCR đọc các ký tự trong khung hình.
5.  **Post-Processing (Quan trọng):** Thuật toán sửa lỗi chuyên biệt cho biển số Việt Nam:
    * 2 ký tự đầu (Mã tỉnh) -> Bắt buộc là **SỐ**.
    * Ký tự thứ 3 (Series) -> Bắt buộc là **CHỮ**.
    * Tự động sửa lỗi phổ biến (VD: Đọc nhầm `O` thành `0`, `B` thành `8`).

### 2. Logic Tính Tiền (Billing Policy)
Hệ thống áp dụng bảng giá lũy tiến:
* **< 10 giờ:** `5.000 VNĐ`
* **10 - 24 giờ:** `10.000 VNĐ`
* **> 24 giờ:** `30.000 VNĐ / ngày` (Làm tròn theo số ngày).

---

## 🚀 Hướng Dẫn Cài Đặt

### Yêu cầu hệ thống
* Python 3.10 trở lên.
* Git.

### Bước 1: Clone dự án
git clone [https://github.com/TEN_GITHUB_CUA_BAN/SmartParkingAI.git](https://github.com/TEN_GITHUB_CUA_BAN/SmartParkingAI.git)
cd SmartParkingAI

### Bước 2: Thiết lập môi trường ảo (Khuyến nghị)Bash# Windows
python -m venv .venv
.venv\Scripts\activate

### Bước 3: Cài đặt thư việnBashpip install -r requirements.txt

### Bước 4: Chạy ứng dụngĐảm bảo file model best.pt đã nằm trong thư mục gốc.Bashpython app.py

### Bước 5: Sử dụngMở trình duyệt web và truy cập: 
👉 https://www.google.com/search?q=http://127.0.0.1:5000📂 

### Cấu Trúc Dự ÁnPlaintextSmartParkingAI/
├── .venv/                 # Môi trường ảo Python (Virtual Environment)
├── static/
│   └── parked_images/     # Kho ảnh: Lưu hình ảnh xe khi Check-in
├── templates/
│   └── index.html         # Frontend: Giao diện người dùng (Dashboard)
├── app.py                 # Backend: Mã nguồn chính (Server & AI Logic)
├── best.pt                # Model: File trọng số YOLOv8 đã train
├── parking_data.json      # Database: Lưu dữ liệu xe và lịch sử
├── requirements.txt       # Config: Danh sách thư viện phụ thuộc
└── README.md              # Document: Tài liệu dự án


<i>Đồ án môn học: Trí Tuệ Nhân Tạo & Thị Giác Máy Tính</i></div>