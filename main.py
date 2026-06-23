import os
import cv2
import numpy as np
import threading
import queue
import time
import streamlit as st
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
import matplotlib.pyplot as plt

# --- CẤU HÌNH GIAO DIỆN WEB ---
st.set_page_config(page_title="Hệ Thống Phân Loại Trái Cây", layout="wide")
st.title("🍎 Hệ Thống Phân Loại Trái Cây")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
FONT_PATH = r"E:\minecraft-f2d-v1-42.otf"
try:
    font_mc = ImageFont.truetype(FONT_PATH, size=22)
except OSError:
    font_mc = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", size=22)

from database import Database
from Dectect import AppleCounter

@st.cache_resource
def init_db_and_model():
    db = Database()
    model = YOLO(r"D:\UTH\Python_Languae\Impotant_Project\may_scan_do_\train-29\weights\best.pt")
    return db, model

db, model = init_db_and_model()

# --- KHỞI TẠO CÁC BIẾN TRẠNG THÁI (SESSION STATE) ---
if "session_id" not in st.session_state:
    st.session_state.session_id = AppleCounter.start_session(db)
    st.session_state.count_apple = 0
    st.session_state.count_green_apple = 0
    st.session_state.count_Moldy_apple = 0
    st.session_state.tracked_ids = set()
    st.session_state.cam_running = False  
    st.session_state.show_report = False  

st.sidebar.markdown(f"### 🕒 Session ID: `#{st.session_state.session_id}`")

# --- LỚP ĐỌC CAMERA ĐA LUỒNG (THREADING) ---
class VideoStreamWidget:
    def __init__(self, src=0):
        # src=0 là webcam mặc định. Đổi thành 1 hoặc link RTSP nếu cần
        self.capture = cv2.VideoCapture(src)
        self.frame_queue = queue.Queue(maxsize=2) 
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            if self.capture.isOpened():
                (status, frame) = self.capture.read()
                if status:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.frame_queue.put(frame)
                else:
                    self.stopped = True
            else:
                self.stopped = True
            time.sleep(0.01) # Tránh CPU chạy quá tải ở luồng đọc ảnh

    def read(self):
        if not self.frame_queue.empty():
            return True, self.frame_queue.get()
        return False, None

    def stop(self):
        self.stopped = True
        time.sleep(0.1)
        if self.capture.isOpened():
            self.capture.release()

def draw_minecraft_text(frame, texts):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    for text, x, y, color in texts:
        draw.text((x + 2, y + 2), text, font=font_mc, fill=(0, 0, 0))
        draw.text((x, y), text, font=font_mc, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- THIẾT KẾ GIAO DIỆN CỘT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎥 Luồng Camera Real-time")
    frame_placeholder = st.empty() 
    report_placeholder = st.container()

    if not st.session_state.cam_running and not st.session_state.show_report:
        frame_placeholder.info("Camera đang TẮT. Hãy nhấn nút '🟢 Bật Camera và Chạy' ở cột bên phải để bắt đầu.")

with col2:
    st.subheader("⚙️ Điều Khiển & Thống Kê")
    
    # 1. BỘ ĐIỀU KHIỂN NÚT BẤM (Đã sửa lỗi kẹt luồng phần cứng)
    if st.session_state.cam_running:
        if st.button("🔴 Tắt Camera & Xem báo cáo", use_container_width=True):
            st.session_state.cam_running = False
            st.session_state.show_report = True
            st.rerun()
    else:
        if st.button("🟢 Bật Camera và Chạy", use_container_width=True, type="primary"):
            st.session_state.cam_running = True
            st.session_state.show_report = False
            time.sleep(0.5) # Tạo độ trễ ngắn cho camera vật lý kịp mở mắt ổn định trước khi reload
            st.rerun()

    # Khung hiển thị biểu đồ cột
    chart_placeholder = st.empty()
    st.markdown("---")
    
    # 2. NÚT XÓA BẢNG (RESET SỐ LIỆU)
    if st.button("🗑️ Xóa Bảng (Reset số liệu đếm)", use_container_width=True):
        st.session_state.count_apple = 0
        st.session_state.count_green_apple = 0
        st.session_state.count_Moldy_apple = 0
        st.session_state.tracked_ids = set()
        st.session_state.show_report = False 
        st.toast("🔄 Đã reset toàn bộ số liệu về 0!")
        st.rerun()

    # 3. NÚT KẾT THÚC CA
    if st.button("🛑 Kết thúc ca làm việc (Đóng Session)", use_container_width=True):
        AppleCounter.end_session(
            db, st.session_state.session_id, 
            st.session_state.count_apple, st.session_state.count_green_apple, st.session_state.count_Moldy_apple
        )
        st.success(f"🎉 Đã kết thúc và lưu Session #{st.session_state.session_id}!")

# --- HÀM VẼ BIỂU ĐỒ ---
def render_chart():
    fig, ax = plt.subplots(figsize=(5, 3.5))
    loai_tao = ["Táo Đỏ", "Táo Xanh", "Táo Hư"]
    so_luong = [st.session_state.count_apple, st.session_state.count_green_apple, st.session_state.count_Moldy_apple]
    ax.bar(loai_tao, so_luong, color=["red", "green", "gray"])
    ax.set_ylabel("Số lượng")
    chart_placeholder.pyplot(fig)
    plt.close(fig)

# Luôn cập nhật đồ thị trạng thái tĩnh lên web
render_chart()

# --- HIỂN THỊ BÁO CÁO SAU KHI TẮT CAM ---
if st.session_state.show_report and not st.session_state.cam_running:
    frame_placeholder.empty() 
    with report_placeholder:
        st.markdown("### 📊 KẾT QUẢ SẢN LƯỢNG ĐÃ XUẤT TRONG CA LÀM VIỆC")
        m1, m2, m3 = st.columns(3)
        m1.metric(label="🍎 Táo Đỏ đã xuất", value=f"{st.session_state.count_apple} quả")
        m2.metric(label="🍏 Táo Xanh đã xuất", value=f"{st.session_state.count_green_apple} quả")
        m3.metric(label="🤢 Táo Hư đã loại bỏ", value=f"{st.session_state.count_Moldy_apple} quả")
        
        tong_cong = st.session_state.count_apple + st.session_state.count_green_apple + st.session_state.count_Moldy_apple
        st.info(f"Tổng số lượng sản phẩm đi qua băng chuyền: **{tong_cong}** quả.")

# --- VÒNG LẶP LIÊN TỤC KHI CAMERA ĐƯỢC KÍCH HOẠT ---
if st.session_state.cam_running:
    # Khởi tạo luồng camera độc lập
    video_stream = VideoStreamWidget(src=0).start()
    
    while st.session_state.cam_running:
        success, frame = video_stream.read()
        if not success:
            time.sleep(0.01)
            continue 

        # Xử lý nhận diện qua mô hình YOLOv12/v8
        results = model.track(
            frame, persist=True, stream=True, conf=0.90, imgsz=480, 
            data=r"D:\UTH\Python_Languae\Impotant_Project\may_scan_do_\Phan_loai_trai_cay-8\data.yaml"
        )

        for r in results:
            if r.boxes is not None and r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.cpu().numpy().astype(int)
                clss = r.boxes.cls.cpu().numpy().astype(int)
                class_names = r.names

                for box, track_id, cls in zip(boxes, ids, clss):
                    name = class_names.get(cls, "")

                    if track_id not in st.session_state.tracked_ids:
                        product_type = None
                        if name == "Apple":
                            st.session_state.count_apple += 1
                            product_type = "Apple"
                        elif name == "green_apple":
                            st.session_state.count_green_apple += 1
                            product_type = "green_apple"
                        elif name == "Moldy_apple":
                            st.session_state.count_Moldy_apple += 1
                            product_type = "Moldy_apple"
                        
                        if product_type:
                            st.session_state.tracked_ids.add(track_id)
                            detection = AppleCounter(st.session_state.session_id, int(track_id), product_type)
                            detection.save_detection(db)

        # Vẽ giao diện nhãn hộp và chữ phong cách Minecraft lên khung ảnh
        frame = r.plot()
        frame = draw_minecraft_text(frame, [
            (f"Tao Do: {st.session_state.count_apple}", 20, 20, (255, 0, 0)),
            (f"Tao Xanh: {st.session_state.count_green_apple}", 20, 55, (0, 255, 0)),
            (f"Tao Hu: {st.session_state.count_Moldy_apple}", 20, 90, (255, 255, 0)),
        ])

        # Kết xuất ảnh lên giao diện Web Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Đồng bộ biểu đồ cột thời gian thực
        render_chart()

    # Thu hồi và đóng camera khi vòng lặp dừng
    video_stream.stop()
