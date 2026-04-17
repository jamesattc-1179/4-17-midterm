import cv2
from deepface import DeepFace

def start_detection():
    # 1. 初始化攝像頭
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("錯誤：無法開啟攝像頭")
        return

    print("程式啟動中，按 'q' 鍵退出...")

    while True:
        # 讀取每一幀影像
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # 2. 使用 DeepFace 進行分析
            # actions 包含：年齡、性別、情緒
            # enforce_detection=False 可避免沒抓到臉時程式崩潰
            results = DeepFace.analyze(frame, 
                                      actions=['age', 'gender', 'emotion'],
                                      enforce_detection=False,
                                      detector_backend='opencv')

            # 3. 處理辨識結果並繪製標籤
            face_count = len(results)
            for res in results:
                # 取得臉部座標
                x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
                
                # 取得辨識資訊
                gender = res['dominant_gender']
                age = res['age']
                emotion = res['dominant_emotion']

                # 繪製方框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # 建立文字標籤
                label = f"{gender}, {age}y, {emotion}"
                cv2.putText(frame, label, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 4. 顯示總人數
            cv2.putText(frame, f"People Count: {face_count}", (20, 40), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 2)

        except Exception as e:
            print(f"分析過程發生異常: {e}")

        # 顯示影像畫面
        cv2.imshow('AI Real-time Recognition', frame)

        # 按 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_detection()