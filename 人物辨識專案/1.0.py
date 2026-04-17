import subprocess
import sys
import numpy as np

# 1. 自動檢查並安裝遺漏的庫
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
        # 隱藏 tkinter 主視窗
        self.root = tk.Tk()
        self.root.withdraw()

    def analyze_face(self, frame):
        """核心辨識邏輯：處理影像並繪製標籤"""
        try:
            results = DeepFace.analyze(frame, 
                                      actions=['age', 'gender', 'emotion'],
                                      enforce_detection=False,
                                      detector_backend='opencv')
            
            for res in results:
                x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
                label = f"{res['dominant_gender']}, {res['age']}y, {res['dominant_emotion']}"
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Count: {len(results)}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
            return frame
        except Exception as e:
            print(f"辨識出錯: {e}")
            return frame

    def process_image(self):
        """處理圖片模式 (支援中文路徑)"""
        file_path = filedialog.askopenfilename(title="選擇圖片", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if file_path:
            # 解決 OpenCV 不支援中文路徑的問題
            img_array = np.fromfile(file_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is not None:
                result_img = self.analyze_face(img)
                cv2.imshow("Image Result (Press any key to close)", result_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("錯誤：無法讀取圖片。")

    def process_video(self, source=0):
        """處理影片或視訊模式"""
        if source is None:
            source = filedialog.askopenfilename(title="選擇影片", filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
            if not source: return

        cap = cv2.VideoCapture(source)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            processed_frame = self.analyze_face(frame)
            cv2.imshow("AI Detection (Press 'q' to quit)", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def main_menu(self):
        """主控選單"""
        while True:
            print("\n--- AI 人臉屬性辨識系統 ---")
            print("1. 辨識圖片檔案")
            print("2. 辨識影片檔案")
            print("3. 開啟即時攝像頭")
            print("4. 退出")
            choice = input("請選擇功能 (1-4): ")

            if choice == '1': self.process_image()
            elif choice == '2': self.process_video(source=None)
            elif choice == '3': self.process_video(source=0)
            elif choice == '4': break
            else: print("無效選擇。")

if __name__ == "__main__":
    app = DetectorApp()
    app.main_menu()