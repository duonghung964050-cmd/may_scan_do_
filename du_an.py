from ultralytics import YOLO
import cv2

# 1. Tải mô hình đã được huấn luyện sẵn (YOLOv8 nano - nhẹ và nhanh)
model = YOLO('yolov8n.pt') 

# 2. Sử dụng Webcam để nhận diện (để dùng ảnh tĩnh, thay 0 bằng đường dẫn file)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. Chạy mô hình trên frame
    results = model(frame)

    # 4. Hiển thị kết quả lên màn hình
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()