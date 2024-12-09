import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

video_yolu = r"C:\Users\BORA\PythonProjeleri\PythonProjeleri\images\sentiment1.mp4"
video = cv2.VideoCapture(video_yolu)

mp_face_mesh = mp.solutions.face_mesh  # mediapipe uygulanması
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

zaman = []
duygu_veri = {"happy": [], "sad": [], "angry": [], "surprise": [], "neutral": []}

frame_sayisi = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(video.get(cv2.CAP_PROP_FPS))
frame_sample_rate = max(1, fps // 2)  # Her yarım saniyede bir işlem
frame_idx = 0

success, frame = video.read()


def duygu_bul(landmarks):
    if landmarks is not None and len(landmarks) > 0:
        sol_goz = landmarks[145][1] - landmarks[159][1]
        sag_goz = landmarks[374][1] - landmarks[386][1]
        agiz = landmarks[13][1] - landmarks[14][1]

        if agiz > 0.02:  # Gülümseme ihtimali
            return "happy"
        elif sol_goz < 0.01 and sag_goz < 0.01:  # Üzgün ihtimali
            return "sad"
        else:
            return "neutral"
    return "neutral"


while success:
    if frame_idx % frame_sample_rate == 0:
        zaman_damgasi = frame_idx / fps
        zaman.append(zaman_damgasi)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sonuc = face_mesh.process(rgb_frame)

        if sonuc.multi_face_landmarks:
            for landmarks in sonuc.multi_face_landmarks:
                # Landmarkları numpy dizisine dönüştürme
                landmark_dizi = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
                duygu = duygu_bul(landmark_dizi)

                # Duygu verilerini kaydetme
                for key in duygu_veri.keys():
                    if key == duygu:
                        duygu_veri[key].append(1)
                    else:
                        duygu_veri[key].append(0)
        else:
            # Yüz bulunamadığında her duygu için 0 ekle
            for key in duygu_veri.keys():
                duygu_veri[key].append(0)

    success, frame = video.read()
    frame_idx += 1

video.release()

plt.figure(figsize=(10, 5))
for duygu, degerler in duygu_veri.items():
    plt.plot(zaman, degerler, label=duygu)
plt.xlabel("Zaman (saniye)")
plt.ylabel("Duygu yoğunluğu")
plt.title("Video Duygu Analizi")
plt.legend()
plt.grid()
plt.show()
