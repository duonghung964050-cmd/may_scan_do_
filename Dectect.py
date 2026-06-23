from datetime import datetime
 
class AppleCounter:
    def __init__(self, session_id, track_id, product_type, detected_at=None):
        self.session_id = session_id
        self.track_id = track_id
        self.product_type = product_type
        self.detected_at = detected_at if detected_at else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
 
    def save_detection(self, db):
        query = """
            INSERT INTO detections (session_id, track_id, product_type, detected_at)
            VALUES (%s, %s, %s, %s)
        """
        params = (self.session_id, self.track_id, self.product_type, self.detected_at)
        db.execute_query(query, params)
 

    @staticmethod
    def start_session(db):
        query = "INSERT INTO sessions (start_time) VALUES (%s)"
        db.execute_query(query, (datetime.now(),))
        result = db.fetch_all("SELECT LAST_INSERT_ID()")
        session_id = result[0][0]
        print(f"Bắt đầu ca làm việc mới. Session ID: #{session_id}")
        return session_id
 
    @staticmethod
    def end_session(db, session_id, count_apple, count_green_apple, count_moldy_apple):
        # Cập nhật thời gian kết thúc ca
        query_end = "UPDATE sessions SET end_time = %s WHERE id = %s"
        db.execute_query(query_end, (datetime.now(), session_id))
 
        query_summary = """
            INSERT INTO session_summary (session_id, count_apple, count_green_apple, count_moldy_apple)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                count_apple = %s,
                count_green_apple = %s,
                count_moldy_apple = %s
        """
        params = (
            session_id, count_apple, count_green_apple, count_moldy_apple,
            count_apple, count_green_apple, count_moldy_apple
        )
        db.execute_query(query_summary, params)
        print(f"Kết thúc ca #{session_id}. Đã lưu tổng kết vào database.")
 


    @staticmethod
    def get_session_report(db, session_id):
        query = """
            SELECT s.id, s.start_time, s.end_time,
                   ss.count_apple, ss.count_green_apple, ss.count_moldy_apple
            FROM sessions s
            LEFT JOIN session_summary ss ON s.id = ss.session_id
            WHERE s.id = %s
        """
        result = db.fetch_all(query, (session_id,))
        return result if result else []
 
    @staticmethod
    def get_all_sessions(db):
        query = """
            SELECT s.id, s.start_time, s.end_time,
                   COALESCE(ss.count_apple, 0),
                   COALESCE(ss.count_green_apple, 0),
                   COALESCE(ss.count_moldy_apple, 0)
            FROM sessions s
            LEFT JOIN session_summary ss ON s.id = ss.session_id
            ORDER BY s.start_time DESC
        """
        result = db.fetch_all(query)
        return result if result else []

    @staticmethod
    def get_total_statistics(db):
        query = """
            SELECT
                SUM(count_apple) AS tong_apple,
                SUM(count_green_apple) AS tong_green_apple,
                SUM(count_moldy_apple) AS tong_moldy_apple,
                COUNT(*) AS tong_so_ca
            FROM session_summary
        """
        result = db.fetch_all(query)
        return result if result else []
 
    # Xem chi tiết từng lần phát hiện trong 1 ca
    @staticmethod
    def get_detections_by_session(db, session_id):
        query = """
            SELECT track_id, product_type, detected_at
            FROM detections
            WHERE session_id = %s
            ORDER BY detected_at ASC
        """
        result = db.fetch_all(query, (session_id,))
        return result if result else []
 
    # Xóa 1 ca theo session_id (xóa cả detections và summary liên quan)
    @staticmethod
    def delete_session(db, session_id):
        db.execute_query("DELETE FROM detections WHERE session_id = %s", (session_id,))
        db.execute_query("DELETE FROM session_summary WHERE session_id = %s", (session_id,))
        db.execute_query("DELETE FROM sessions WHERE id = %s", (session_id,))
        print(f"Đã xóa ca #{session_id} khỏi database.")