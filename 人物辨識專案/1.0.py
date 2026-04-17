import subprocess
import sys
import numpy as np

# 1. 自動安裝依賴庫
def install_dependencies():
    required = {'opencv-python', 'deepface', 'tf-keras'}
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
        """只負責 AI 分析，改用 SSD 引擎減少誤判"""
        try:
            # 將 detector_backend 改為 'ssd'，它比 opencv 精準很多，且比 retinaface 快
            return DeepFace.analyze(frame, 
                                   actions=['age', 'gender', 'emotion'],
                                   enforce_detection=False, 
                                   detector_backend='ssd') # 這裡從 'opencv' 改為 'ssd'
        except Exception as e:
            # print(f"分析錯誤: {e}") # 除錯用
            return []

    def draw_results(self, frame, results):
        """強化版繪圖：確保有偵測到臉就一定會畫框"""
        if not results or len(results) == 0: return frame
        
        img_w = frame.shape[1]
        font_scale = img_w / 1000.0 * 0.7 
        thickness = max(1, int(img_w / 500))
        
        results_sorted = sorted(results, key=lambda x: x['region']['x'])
        occupied_regions = []

        for res in results_sorted:
            # 取得位置
            region = res.get('region', {})
            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
            
            # 安全取得標籤資訊，若無資料則顯示 N/A
            gender = res.get('dominant_gender', 'N/A')
            age = res.get('age', '??')
            emotion = res.get('dominant_emotion', 'N/A')
            label = f"{gender}, {age}y, {emotion}"
            
            # 1. 先畫方框 (保證一定會出現)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness)
            
            # 2. 計算不重疊位置
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            curr_y = y - 10
            while True:
                collision = False
                for (ox, oy, ow, oh) in occupied_regions:
                    if not (x + text_w < ox or x > ox + ow):
                        if abs(curr_y - oy) < (text_h + 10):
                            collision = True; break
                if collision: curr_y -= (text_h + 25)
                else: break
            
            # 紀錄並畫出標籤
            occupied_regions.append((x, curr_y, text_w, text_h))
            cv2.rectangle(frame, (x, curr_y - text_h - 5), (x + text_w, curr_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, label, (x, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        
        # 標註總人數
        cv2.putText(frame, f"Count: {len(results)}", (int(20*font_scale), int(50*font_scale)), 
                    cv2.FONT_HERSHEY_DUPLEX, font_scale*1.2, (255, 0, 0), thickness)
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
            
            # 每 5 幀才進行一次重型運算
            if frame_count % 5 == 0:
                last_results = self.analyze_face(frame)
            
            # 每一幀都負責畫圖 (保持畫面流暢)
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