import subprocess
import sys
import numpy as np

# 1. 自動安裝依賴庫 (確保老師或同學拿去跑時也能自動設定環境)
def install_dependencies():
    required = {'opencv-python', 'deepface', 'tf-keras', 'mediapipe'}
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"偵測到環境缺少 {package}，正在安裝...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_dependencies()

import cv2
import tkinter as tk
from tkinter import filedialog
from deepface import DeepFace

class DetectorApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()

    def analyze_face(self, frame):
        """核心辨識邏輯：使用最靈敏引擎並確保回傳資料"""
        try:
            # 換回 opencv 確保靈敏度，enforce_detection=False 確保不崩潰
            return DeepFace.analyze(frame, 
                                   actions=['age', 'gender', 'emotion'],
                                   enforce_detection=False, 
                                   detector_backend='opencv') 
        except Exception as e:
            return []

    def draw_results(self, frame, results):
        """強化繪圖：過濾雜訊並保證顯示文字"""
        if not results: return frame
        
        img_w = frame.shape[1]
        font_scale = img_w / 1000.0 * 0.7 
        thickness = max(1, int(img_w / 500))
        
        results_sorted = sorted(results, key=lambda x: x.get('region', {}).get('x', 0))
        occupied_regions = []

        for res in results_sorted:
            region = res.get('region', {})
            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
            
            # --- 關鍵優化 1：過濾衣服褶皺 (褶皺通常寬度或高度會小於人臉常態) ---
            # 如果是遠景照片，可以改小一點 (如 40)；近景則維持 60-80
            if w < 50 or h < 50: 
                continue

            # 安全取得資訊，避免 Key 缺失導致沒畫面
            gender = res.get('dominant_gender', 'Processing')
            age = res.get('age', '??')
            emotion = res.get('dominant_emotion', '...')
            label = f"{gender}, {age}y, {emotion}"
            
            # 畫人臉框 (亮綠色)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness)
            
            # --- 關鍵優化 2：動態文字位置校正 ---
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            curr_y = y - 10
            
            # 防止文字出界（如果人在最上方，文字會跑到畫面外）
            if curr_y < 30: curr_y = y + h + 30 
            
            # 防重疊邏輯
            while True:
                collision = False
                for (ox, oy, ow, oh) in occupied_regions:
                    if not (x + text_w < ox or x > ox + ow):
                        if abs(curr_y - oy) < (text_h + 15):
                            collision = True; break
                if collision: curr_y -= (text_h + 25)
                else: break
            
            occupied_regions.append((x, curr_y, text_w, text_h))
            
            # 畫文字底色與標籤 (增加對比度)
            cv2.rectangle(frame, (x, curr_y - text_h - 5), (x + text_w, curr_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, label, (x, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        
        return frame

    def process_image(self):
        """圖片辨識模式"""
        file_path = filedialog.askopenfilename(title="選擇圖片", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if file_path:
            img_array = np.fromfile(file_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is not None:
                results = self.analyze_face(img)
                result_img = self.draw_results(img, results)
                win_name = "Image Result"
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                h, w = result_img.shape[:2]
                cv2.resizeWindow(win_name, 1280, int(h * (1280 / w)) if w > 1280 else h)
                cv2.imshow(win_name, result_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def process_video(self, source=0):
        """影片/攝像頭辨識模式 (每 5 幀辨識一次)"""
        if source is None:
            source = filedialog.askopenfilename(title="選擇影片", filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
            if not source: return
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"\n[錯誤] 無法開啟來源 {source}")
            return

        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
        frame_count = 0
        last_results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_count % 5 == 0:
                last_results = self.analyze_face(frame)
            
            processed = self.draw_results(frame, last_results)
            cv2.imshow("Detection", processed)
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()

    def main_menu(self):
        """主選單"""
        while True:
            print("\n--- AI 人臉屬性辨識系統 (優化版) ---")
            print("1. 辨識圖片檔案\n2. 辨識影片檔案\n3. 開啟即時攝像頭\n4. 退出")
            choice = input("請選擇功能 (1-4): ")
            if choice == '1': self.process_image()
            elif choice == '2': self.process_video(source=None)
            elif choice == '3': self.process_video(source=0)
            elif choice == '4': break

if __name__ == "__main__":
    app = DetectorApp()
    app.main_menu()