import cv2
import numpy as np
import time
from collections import deque
from statistics import mode, StatisticsError
from ultralytics import YOLO
import mediapipe as mp

print("Chargement du Systeme Hybride Haute Précision...")
model_yolo = YOLO('best.pt')

model_path = 'hand_landmarker.task'
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1)

historique_lettres = deque(maxlen=7)
lettre_stable = "..."
confiance_stable = 0.0

lissage = 0.6 
old_x1, old_y1, old_x2, old_y2 = 0, 0, 0, 0

connexions = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),       
    (5,9), (9,10), (10,11), (11,12), (9,13), (13,14), (14,15), (15,16),
    (13,17), (0,17), (17,18), (18,19), (19,20)
]

cap = cv2.VideoCapture(0)
print("Appuyez sur 'q' pour quitter.")

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (640, 480))
        h, w, _ = frame.shape
        
        panel = np.zeros((480, 250, 3), dtype=np.uint8)
        cv2.putText(panel, "TRADUCTION LSF", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.line(panel, (10, 50), (240, 50), (255, 255, 255), 1)

        # --- A. MEDIAPIPE (Localisation) ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        mp_results = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        if mp_results.hand_landmarks:
            hand_landmarks = mp_results.hand_landmarks[0]
            points_pixels, x_coords, y_coords = [], [], []
            
            for landmark in hand_landmarks:
                x, y = int(landmark.x * w), int(landmark.y * h)
                points_pixels.append((x, y))
                x_coords.append(x); y_coords.append(y)

            # Calcul de la zone de la main avec marge
            marge = 40
            x1, y1 = max(0, min(x_coords) - marge), max(0, min(y_coords) - marge)
            x2, y2 = min(w, max(x_coords) + marge), min(h, max(y_coords) + marge)

            # Lissage spatial
            if old_x2 == 0: old_x1, old_y1, old_x2, old_y2 = x1, y1, x2, y2
            else:
                x1 = int(old_x1 * lissage + x1 * (1 - lissage))
                y1 = int(old_y1 * lissage + y1 * (1 - lissage))
                x2 = int(old_x2 * lissage + x2 * (1 - lissage))
                y2 = int(old_y2 * lissage + y2 * (1 - lissage))
            old_x1, old_y1, old_x2, old_y2 = x1, y1, x2, y2

            # --- AMÉLIORATION MAJEURE : LE MASQUE NOIR ---
            # On crée une image totalement noire de la taille de la webcam
            frame_masque = np.zeros((h, w, 3), dtype=np.uint8)
            # On "copie-colle" uniquement la main de la vraie caméra vers l'image noire
            frame_masque[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

            # --- B. YOLO (Sur l'image masquée) ---
            # YOLO regarde la grande image, mais seul la main est éclairée !
            yolo_results = model_yolo.predict(frame_masque, verbose=False, conf=0.25)
            
            meilleure_conf = 0
            lettre_brute = None

            for r in yolo_results:
                if len(r.boxes) > 0:
                    box = r.boxes[0] 
                    conf = float(box.conf[0])
                    if conf > meilleure_conf:
                        meilleure_conf = conf
                        lettre_brute = model_yolo.names[int(box.cls[0])]

            if lettre_brute is not None:
                historique_lettres.append(lettre_brute)
                try:
                    lettre_stable = mode(historique_lettres)
                    confiance_stable = meilleure_conf
                except StatisticsError: pass

            # --- C. INTERFACE ---
            for connexion in connexions:
                cv2.line(frame, points_pixels[connexion[0]], points_pixels[connexion[1]], (255, 255, 255), 2)
            for pt in points_pixels: cv2.circle(frame, pt, 4, (0, 150, 255), -1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
            cv2.putText(frame, lettre_stable, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 100), 2)
            cv2.putText(panel, f"Signe: {lettre_stable}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
            cv2.putText(panel, f"Preci: {confiance_stable:.0%}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        else:
            historique_lettres.clear(); old_x2 = 0; lettre_stable = "..."
            cv2.putText(panel, "Aucune main", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        fenetre_finale = cv2.hconcat([frame, panel])
        
        # Ligne de debug optionnelle : dé-commentez pour voir ce que YOLO voit vraiment
        # cv2.imshow("Ce que YOLO voit", frame_masque) 

        cv2.imshow("Pipeline Langue des Signes PRO", fenetre_finale)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()