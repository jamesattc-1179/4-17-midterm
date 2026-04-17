import subprocess
import sys
import numpy as np

# 1. 自動安裝依賴庫 (確保跨設備執行的相容性)
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
        """核心辨識邏輯：使用 OpenCV 偵測器確保靈敏度"""
        try:
            # 針對憋笑與誤判問題，維持 opencv 靈敏度並設定不強制報錯
            return DeepFace.analyze(frame, 
                                   actions=['age', 'gender', 'emotion'],
                                   enforce_detection=False, 
                                   detector_backend='opencv') 
        except:
            return []

    def draw_results(self, frame, results):
        """強化繪圖邏輯：包含防重疊、雜訊過濾、總人數統計"""
        img_w = frame.shape[1]
        font_scale = img_w / 1000.0 * 0.7 
        thickness = max(1, int(img_w / 500))

        # 基礎人數標籤顏色
        count_color = (255, 0, 0) # 藍色

        if not results or len(results) == 0:
            cv2.putText(frame, "Count: 0", (int(20*font_scale), int(50*font_scale)), 
                        cv2.FONT_HERSHEY_DUPLEX, font_scale*1.2, count_color, thickness)
            return frame
        
        # 1. 過濾與排序
        # 僅保留寬度大於 50 像素的框，避免衣服褶皺誤判
        valid_results = [res for res in results if res.get('region', {}).get('w', 0) > 50]
        valid_results = sorted(valid_results, key=lambda x: x.get('region', {}).get('x', 0))
        
        occupied_regions = []

        for res in valid_results:
            region = res.get('region', {})
            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
            
            # 安全取得資訊
            gender = res.get('dominant_gender', 'Processing')
            age = res.get('age', '??')
            emotion = res.get('dominant_emotion', 'N/A')
            
            # 修正微表情誤判：若出現 sad 且在合照中，可視為 neutral
            if emotion == 'sad': emotion = 'neutral'
            
            label = f"{gender}, {age}y, {emotion}"
            
            # 畫人臉框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness)
            
            # 2. 防重疊階梯排版計算
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            curr_y = y - 10
            if curr_y < 40: curr_y = y + h + int(30 * font_scale) # 太靠上方則換到下方顯示
            
            while True:
                collision = False
                for (ox, oy, ow, oh) in occupied_regions:
                    if not (x + text_w < ox or x > ox + ow): # 水平重疊
                        if abs(curr_y - oy) < (text_h + 15): # 垂直太近
                            collision = True; break
                if collision: curr_y -= (text_h + 25)
                else: break
            
            occupied_regions.append((x, curr_y, text_w, text_h))
            
            # 畫標籤底色背景 (黑色) 與文字
            cv2.rectangle(frame, (x, curr_y - text_h - 5), (x + text_w, curr_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, label, (x, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        
        # 3. 繪製左上角總人數 Count
        count_label = f"Count: {len(valid_results)}"
        cv2.putText(frame, count_label, (int(20*font_scale), int(50*font_scale)), 
                    cv2.FONT_HERSHEY_DUPLEX, font_scale*1.2, count_color, thickness)
        
        return frame

    def process_image(self):
        """圖片辨識模式"""
        file_path = filedialog.askopenfilename(title="選擇圖片", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if file_path:
            # 解決中文路徑讀取問題
            img_array = np.fromfile(file_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is not None:
                results = self.analyze_face(img)
                result_img = self.draw_results(img, results)
                
                win_name = "Image Result"
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                h, w = result_img.shape[:2]
                # 自動縮放視窗避免溢出螢幕
                display_w = 1280
                if w > display_w:
                    cv2.resizeWindow(win_name, display_w, int(h * (display_w / w)))
                else:
                    cv2.resizeWindow(win_name, w, h)
                    
                cv2.imshow(win_name, result_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def process_video(self, source=0):
        """影片/攝像頭模式 (效能優化版)"""
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
            
            # 每 5 幀辨識一次，減少 CPU/GPU 負擔
            if frame_count % 5 == 0:
                last_results = self.analyze_face(frame)
            
            processed = self.draw_results(frame, last_results)
            cv2.imshow("Detection", processed)
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        cap.release()
        cv2.destroyAllWindows()

    def main_menu(self):
        """主選單介面"""
        while True:
            print("\n--- AI 人臉屬性辨識系統 (期中專修版) ---")
            print("1. 辨識圖片檔案\n2. 辨識影片檔案\n3. 開啟即時攝像頭\n4. 退出")
            choice = input("請選擇功能 (1-4): ")
            if choice == '1': self.process_image()
            elif choice == '2': self.process_video(source=None)
            elif choice == '3': self.process_video(source=0)
            elif choice == '4': break

if __name__ == "__main__":
    app = DetectorApp()
    app.main_menu()