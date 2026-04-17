import subprocess
import sys
import numpy as np

# 1. 自動檢查並安裝遺漏的庫 (對應作業 Debug 歷程)
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
        """核心辨識邏輯：包含動態縮放與防重疊排版"""
        try:
            img_w = frame.shape[1]
            font_scale = img_w / 1000.0 * 0.7 
            thickness = max(1, int(img_w / 500)) 

            results = DeepFace.analyze(frame, 
                                      actions=['age', 'gender', 'emotion'],
                                      enforce_detection=False,
                                      detector_backend='opencv')
            
            # 防重疊排序
            results = sorted(results, key=lambda x: x['region']['y'])
            
            for i, res in enumerate(results):
                x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
                label = f"{res['dominant_gender']}, {res['age']}y, {res['dominant_emotion']}"
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness)
                
                # 樓層式偏移排版
                y_offset = int(15 * font_scale)
                if i % 2 == 1:
                    y_offset += int(40 * font_scale)
                
                # 繪製文字底色背景
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(frame, (x, y - y_offset - text_h), (x + text_w, y - y_offset + 5), (0, 0, 0), -1)
                
                cv2.putText(frame, label, (x, y - y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
            
            cv2.putText(frame, f"Count: {len(results)}", (int(20 * font_scale), int(50 * font_scale)), 
                        cv2.FONT_HERSHEY_DUPLEX, font_scale * 1.2, (255, 0, 0), thickness)
            return frame
        except Exception as e:
            return frame

    def process_image(self):
        """處理圖片模式 (支援中文與自動縮放)"""
        file_path = filedialog.askopenfilename(title="選擇圖片", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if file_path:
            img_array = np.fromfile(file_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is not None:
                result_img = self.analyze_face(img)
                win_name = "Image Result"
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                
                h, w = result_img.shape[:2]
                if w > 1280:
                    cv2.resizeWindow(win_name, 1280, int(h * (1280 / w)))
                else:
                    cv2.resizeWindow(win_name, w, h)

                cv2.imshow(win_name, result_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def process_video(self, source=0):
        """處理影片或視訊模式"""
        if source is None:
            source = filedialog.askopenfilename(title="選擇影片", filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
            if not source: return

        cap = cv2.VideoCapture(source)
        cv2.namedWindow("AI Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AI Detection", 1280, 720)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            processed_frame = self.analyze_face(frame)
            cv2.imshow("AI Detection", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()

    def main_menu(self):
        """主選單介面"""
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

if __name__ == "__main__":
    app = DetectorApp()
    app.main_menu()