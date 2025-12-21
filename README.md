# Smart Parking AI - Hệ Thống Quản Lý Bãi Đỗ Xe Thông Minh

> **Đồ án môn học: Trí Tuệ Nhân Tạo (AI)** > **Phiên bản:** 1.0.0  
> **Công nghệ:** Python, Flask, YOLOv8, EasyOCR, OpenCV

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-orange)

## Giới thiệu
Đây là hệ thống quản lý bãi đỗ xe tự động sử dụng công nghệ Thị giác máy tính (Computer Vision). Hệ thống giúp tự động nhận diện biển số xe máy/ô tô khi vào và ra, tính toán thời gian gửi và chi phí một cách chính xác, minh bạch.

Dự án được xây dựng trên nền tảng Web (Flask), cho phép triển khai dễ dàng và truy cập qua trình duyệt trên cả máy tính và điện thoại.

## Tính năng nổi bật
- **Nhận diện Real-time:** Quét biển số trực tiếp từ Webcam hoặc Camera IP.
- **Hỗ trợ đa nguồn:** Cho phép upload ảnh tĩnh hoặc video có sẵn để xử lý.
- **AI Mạnh mẽ:** - Sử dụng **YOLOv8** để phát hiện vùng biển số.
  - Sử dụng **EasyOCR** kết hợp thuật toán xử lý ảnh nâng cao (Gamma Correction, Sharpening) để đọc chữ số chính xác.
  - Logic tự động sửa lỗi chính tả biển số Việt Nam (VD: đọc nhầm `8` thành `B`, `0` thành `D`).
- **Tính tiền tự động:** Tự động tính toán thời gian gửi và số tiền khách phải trả.
- **Thống kê:** Hiển thị số lượng xe đang gửi và tổng doanh thu theo thời gian thực.
- **Giao diện Web:** Hiện đại, dễ sử dụng, tương thích Mobile.

Cấu trúc dự án PlaintextSmartParkingAI/
├── app.py                 # File code chính (Backend Flask)
├── best.pt                # Model AI (YOLOv8) đã train
├── parking_data.json      # Cơ sở dữ liệu JSON (Lưu lịch sử xe)
├── requirements.txt       # Danh sách thư viện
├── README.md              # Hướng dẫn sử dụng
├── static/                # Thư mục chứa file tĩnh
│   └── parked_images/     # Ảnh chụp xe khi Check-in
└── templates/
    └── index.html         # Giao diện Web (Frontend)

Hướng dẫn sử dụng nhanhXe Vào (Check-in):
   1. Đưa xe vào trước Camera (hoặc tải ảnh lên).
   2. Nhấn nút "XE VÀO".Hệ thống lưu biển số, giờ vào và chụp ảnh lại.
   3. XeRa (Check-out):Khi xe ra, nhấn nút "XE RA".Hệ thống tính toán thời gian và hiện số tiền cần thu.
   4. Xe được xóa khỏi danh sách đang gửi.
   5. Chuyển chế độ:Dùng nút "Tải ảnh lên" để test với ảnh chụp sẵn.Dùng nút "Bật Camera" để quay lại chế độ Webcam.