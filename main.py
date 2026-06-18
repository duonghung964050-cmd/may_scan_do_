import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

from PIL import ImageFont, ImageDraw, Image
import numpy as np

FONT_PATH = r"D:\UTH\Python_Languae\Impotant_Project\may_scan_do_\MinecraftRegular-Bmg3.otf"
font_mc = ImageFont.truetype(FONT_PATH, size=22)  # Chỉnh size tùy thích

def draw_minecraft_text(frame, texts):
    """
    texts: list các tuple (text, x, y, color_rgb)
    """
    # Chuyển frame OpenCV (BGR) → PIL (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    for text, x, y, color in texts:
        # Vẽ shadow (bóng đen) lệch 2px cho giống Minecraft
        draw.text((x + 2, y + 2), text, font=font_mc, fill=(0, 0, 0))
        # Vẽ chữ chính
        draw.text((x, y), text, font=font_mc, fill=color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

model = YOLO("D:/UTH/Python_Languae/Impotant_Project/may_scan_do_/train-23/weights/best.pt")

# Khởi tạo biến đếm sản phẩm và danh sách lưu trữ ID để tránh đếm trùng
count_xoai = 0
count_tao = 0
count_chanh = 0
tracked_ids = set() # Nơi lưu các ID đã đi qua băng chuyền

cap = cv2.VideoCapture(0)
print("Hệ thống Đếm Sản Phẩm Băng Chuyền đang chạy...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 🚨 THAY THẾ model() BẰNG model.track() ĐỂ AI TỰ CẤP ID CHO TỪNG QUẢ
    # persist=True giúp AI nhớ ID của quả đó khi nó di chuyển trên băng chuyền
    results = model.track(
        frame, 
        persist=True, 
        stream=True, 
        conf=0.40, 
        imgsz=640,
        data="D:/training/Phan_loai_trai_cay-4/data.yaml"
    )

    for r in results:
        # Kiểm tra xem khung hình hiện tại có bắt được vật thể nào có ID không
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            clss = r.boxes.cls.cpu().numpy().astype(int)
            
            # Lấy danh sách tên nhãn từ bộ não (0: xoai, 1: tao...)
            class_names = r.names

            # Vòng lặp quét qua từng vật thể đang xuất hiện trên màn hình
            for box, track_id, cls in zip(boxes, ids, clss):
                name = class_names.get(cls, "")

                # Nếu ID này là hoàn toàn mới (chưa từng đi qua băng chuyền trước đây)
                if track_id not in tracked_ids:
                    if name == "xoai":
                        count_xoai += 1
                        tracked_ids.add(track_id) # Đánh dấu đã đếm quả xoài này
                        print(f"-> Phát hiện 1 quả Xoài mới! Tổng số Xoài: {count_xoai}")
                    elif name == "apple":
                        count_tao += 1
                        tracked_ids.add(track_id) # Đánh dấu đã đếm quả táo này
                        print(f"-> Phát hiện 1 quả Táo mới! Tổng số Táo: {count_tao}")
                    elif name == "chanh":
                        count_chanh += 1
                        tracked_ids.add(track_id)
                        print(f"-> Phát hiện 1 quả Chanh mới! Tổng số Chanh: {count_chanh}")

        # Giữ nguyên cấu trúc tự động vẽ khung và tên của bạn cho ngắn gọn
        frame = r.plot() 

    # 📊 VẼ BẢNG THỐNG KÊ SỐ LƯỢNG LÊN MÀN HÌNH CAMERA
    # Tạo một hộp nền đen góc trên bên trái để hiển thị số lượng cho chuyên nghiệp
    # Thay 3 dòng cv2.putText cũ bằng đoạn này
    frame = draw_minecraft_text(frame, [
        (f"Xoai:  {count_xoai}",  20, 20,  (255, 255,   0)),  # Vàng
        (f"Tao:   {count_tao}",   20, 55,  (255,  80,  80)),  # Đỏ
        (f"Chanh: {count_chanh}", 20, 90,  ( 80, 255,  80)),  # Xanh lá
    ])

    # Hiển thị cửa sổ camera sản xuất
    cv2.imshow("YOLOv12 - Bang Chuyen San Xuat Tu Dong", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
print(f"--- BÁO CÁO CA LÀM VIỆC ---")
print(f"Tổng số lượng xoài đã đóng gói: {count_xoai}")
print(f"Tổng số lượng táo đã đóng gói: {count_tao}")
print(f"Tổng số lượng chanh đã đóng gói: {count_chanh}")