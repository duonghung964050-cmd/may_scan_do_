import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

# 1. THAY ĐỔI ĐƯỜNG DẪN TỚI THƯ MỤC TRAIN MỚI NHẤT VỪA CHẠY XONG (Có chứa quả xoài)
# Bạn hãy vào thư mục D:/luffy/runs/detect/ xem thư mục nào có số TO NHẤT thì điền vào đây thay cho train-12 nhé!
model = YOLO("D:/luffy/runs/detect/train-12/weights/best.pt")

# 2. Mở Webcam mặc định
cap = cv2.VideoCapture(0)

print("Đang chạy camera nhận diện Luffy và Xoài...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Giữ mức conf ổn định ở tầm 0.15 đến 0.25 để lọc bớt nhiễu nhận diện nhầm
    results = model(frame, stream=True, conf=0.50, imgsz=640)

    # Dùng hàm vẽ tự động r.plot() để hiển thị đúng nhãn và khung hình từ bộ não mới
    for r in results:
        frame = r.plot() 

    # Hiển thị cửa sổ camera
    cv2.imshow("YOLOv12 - Chuan Ten San Pham", frame)

    # Nhấn Q để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
