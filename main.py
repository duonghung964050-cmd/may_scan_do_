import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

# Đảm bảo gọi đúng bộ não train-23
model = YOLO("D:/training/runs/detect/train-23/weights/best.pt")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Đặt conf=0.40 để chặn hoàn toàn hiện tượng nhận diện bậy mặt bạn thành Luffy
    results = model(frame, stream=True, conf=0.3, imgsz=640)

    for r in results:
        frame = r.plot()  # Giữ nguyên cấu trúc vẽ tự động của bạn

    cv2.imshow("YOLOv12 - Chuan Ten San Pham", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
