import cv2
from fer import FER
import matplotlib.pyplot as plt

dizin = r"C:\Users\BORA\PythonProjeleri\PythonProjeleri\images\duygu.mp4"
video = cv2.VideoCapture(dizin)

frame_sayim = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(video.get(cv2.CAP_PROP_FPS))
sure = frame_sayim / fps
frame_sample_rate = max(1, fps // 2)  #  yarım saniyede bir kare

# fer
detector = FER(mtcnn=True) # fer aracı yüz tespitinin işlemeri için MTCNN ağını kulanıyor

duygu_veri = []
zaman = []
frame_idx = 0
success, frame = video.read()

while success:             # duygusal analizdöngüsü
    if frame_idx % frame_sample_rate == 0:
        # zaman damgası
        timestamp = frame_idx / fps
        zaman.append(timestamp)

        # karedeki yüzleri fer ile analiz etme
        results = detector.detect_emotions(frame)
        frame_emotions = []
        for result in results:
            #  yüz için en baskın duygu
            dominant_emotion = max(result["emotions"], key=result["emotions"].get)
            frame_emotions.append(dominant_emotion)


        duygu_veri.append({"timestamp": timestamp, "emotions": frame_emotions})

    success, frame = video.read()
    frame_idx += 1

video.release()

emotions_over_time = {emotion: [] for emotion in ["happy", "sad", "angry", "surprise", "neutral", "fear", "disgust"]}
for entry in duygu_veri:
    frame_emotions = entry["emotions"]
    for emotion in emotions_over_time:
        emotions_over_time[emotion].append(frame_emotions.count(emotion))

plt.figure(figsize=(12, 6))
for emotion, values in emotions_over_time.items():
    plt.plot(zaman, values, label=emotion)
plt.xlabel("Zaman (saniye)")
plt.ylabel("Duygu yoğunluğu")
plt.title("Video Duygu Analizi")
plt.legend()
plt.grid()
plt.show()





