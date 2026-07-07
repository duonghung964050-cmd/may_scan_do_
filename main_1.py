import cv2
import os
import matplotlib.pyplot as plt

from requests import session
from database import Database
from Dectect import AppleCounter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

from PIL import ImageFont, ImageDraw, Image
import numpy as np

FONT_PATH = r"D:\may_phan_loai_trai_cay\minecraft-f2d-v1-42.otf"
font_mc = ImageFont.truetype(FONT_PATH, size=22)

def draw_minecraft_text(frame, texts):
    """
    texts: list các tuple (text, x, y, color_rgb)
    """
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    for text, x, y, color in texts:
        draw.text((x + 2, y + 2), text, font=font_mc, fill=(0, 0, 0))
        draw.text((x, y), text, font=font_mc, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ==================== KẾT NỐI DATABASE ====================
db = Database()
session_id = AppleCounter.start_session(db)
# ===========================================================

model = YOLO("D:/training/runs/detect/train-29/weights/best.pt")

count_apple      = 0
count_green_apple = 0
count_Moldy_apple = 0
tracked_ids = set()

cap = cv2.VideoCapture(0)
print("Hệ thống Đếm Sản Phẩm Băng Chuyền đang chạy...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(
        frame,
        persist=True,
        stream=True,
        conf=0.90,
        imgsz=640,
        data="D:/training/Phan_loai_trai_cay-8/data.yaml"
    )

    for r in results:
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            clss = r.boxes.cls.cpu().numpy().astype(int)
            class_names = r.names

            for box, track_id, cls in zip(boxes, ids, clss):
                name = class_names.get(cls, "")

                if track_id not in tracked_ids:
                    product_type = None

                    if name == "Apple":
                        count_apple += 1
                        product_type = "Apple"
                        print(f"-> Phát hiện 1 quả Táo mới! Tổng: {count_apple}")

                    elif name == "green_apple":
                        count_green_apple += 1
                        product_type = "green_apple"
                        print(f"-> Phát hiện 1 quả Táo Xanh mới! Tổng: {count_green_apple}")

                    elif name == "Moldy_apple":
                        count_Moldy_apple += 1
                        product_type = "Moldy_apple"
                        print(f"-> Phát hiện 1 quả Táo Hư mới! Tổng: {count_Moldy_apple}")

                    
                    if product_type:
                        tracked_ids.add(track_id)
                        detection = AppleCounter(session_id, int(track_id), product_type)
                        detection.save_detection(db)
                    

        frame = r.plot()

    frame = draw_minecraft_text(frame, [
        (f"Táo Đỏ: {count_apple}",        20, 20, (255, 0, 0)),
        (f"Táo Xanh: {count_green_apple}",  20, 55, (0, 255, 0)),
        (f"Táo Hư: {count_Moldy_apple}",  20, 90, (255, 255, 0)),
    ])

    cv2.imshow("YOLOv12 - Bang Chuyen San Xuat Tu Dong", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

AppleCounter.end_session(db, session_id, count_apple, count_green_apple, count_Moldy_apple)


print(f"\n--- BÁO CÁO CA LÀM VIỆC (Session #{session_id}) ---")
print(f"Tổng số lượng Táo Đỏ đã đóng gói    : {count_apple}")
print(f"Tổng số lượng Táo Xanh đã đóng gói: {count_green_apple}")
print(f"Tổng số lượng Táo Hư đã đóng gói  : {count_Moldy_apple}")


# Tạo dữ liệu cho biểu đồ
loai_tao = ["Táo Đỏ", "Táo Xanh", "Táo Hư"]
so_luong = [count_apple, count_green_apple, count_Moldy_apple]
mau_sac = ["red", "green", "gray"]

# In báo cáo ra màn hình Terminal
print(f"--- BÁO CÁO CA LÀM VIỆC (Session #{session_id}) ---")
print(f"Tổng số lượng Táo Đỏ đã đóng gói   : {count_apple}")
print(f"Tổng số lượng Táo Xanh đã đóng gói  : {count_green_apple}")
print(f"Tổng số lượng Táo Hư đã đóng gói    : {count_Moldy_apple}")

# Vẽ biểu đồ cột
plt.figure(figsize=(8, 5))
cot = plt.bar(loai_tao, so_luong, color=mau_sac)

# Tiêu đề và nhãn
plt.title(f"BÁO CÁO CA LÀM VIỆC - SESSION #{session_id}", fontsize=14, fontweight="bold")
plt.xlabel("Loại táo")
plt.ylabel("Số lượng đã đóng gói")

# Hiện số lượng trên đầu từng cột
for c in cot:
    chieu_cao = c.get_height()
    plt.text(
        c.get_x() + c.get_width() / 2,
        chieu_cao + 0.2,
        str(int(chieu_cao)),
        ha="center",
        fontsize=12
    )

# Canh trục Y cho đẹp
plt.ylim(0, max(so_luong) + 3)

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()

loai_tao = ["Táo Đỏ", "Táo Xanh", "Táo Hư"]
so_luong = [count_apple, count_green_apple, count_Moldy_apple]
mau_sac = ["red", "green", "gray"]


# In báo cáo ra màn hình Terminal
print(f"--- BÁO CÁO CA LÀM VIỆC (Session #{session_id}) ---")
print(f"Tổng số lượng Táo Đỏ đã đóng gói   : {count_apple}")
print(f"Tổng số lượng Táo Xanh đã đóng gói  : {count_green_apple}")
print(f"Tổng số lượng Táo Hư đã đóng gói    : {count_Moldy_apple}")


# Vẽ biểu đồ cột
plt.figure(figsize=(8, 5))
cot = plt.bar(loai_tao, so_luong, color=mau_sac)


# Tiêu đề và nhãn
plt.title(f"BÁO CÁO CA LÀM VIỆC - SESSION #{session_id}", fontsize=14, fontweight="bold")
plt.xlabel("Loại táo")
plt.ylabel("Số lượng đã đóng gói")


# Hiện số lượng trên đầu từng cột
for c in cot:
    chieu_cao = c.get_height()
    plt.text(
        c.get_x() + c.get_width() / 2,
        chieu_cao + 0.2,
        str(int(chieu_cao)),
        ha="center",
        fontsize=12
    )


# Canh trục Y cho đẹp
plt.ylim(0, max(so_luong) + 3)


# Hiển thị biểu đồ
plt.tight_layout()
plt.show()

