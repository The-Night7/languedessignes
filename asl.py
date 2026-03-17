import cv2
import numpy as np
import time
from collections import deque
from statistics import mode, StatisticsError
from ultralytics import YOLO
import mediapipe as mp

# --- 1. CHARGEMENT DES DEUX IA ---
# YOLO (L'Intelligence qui traduit la lettre)
model = YOLO('best.pt')

# MediaPipe (Le traceur qui dessine l'anatomie)
model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2)

# --- 2. CONFIGURATION VISUELLE ET MÉMOIRE ---
# Dictionnaire des os de la main pour MediaPipe
connexions = [
    (0,1), (1,2), (2,3), (3,4),       # Pouce
    (0,5), (5,6), (6,7), (7,8),       # Index
    (5,9), (9,10), (10,11), (11,12),  # Majeur
    (9,13), (13,14), (14,15), (15,16),# Annulaire
    (13,17), (0,17), (17,18), (18,19), (19,20) # Auriculaire
]

historique_lettres = deque(maxlen=5)
lettre_stable = "..."

cap = cv2.VideoCapture(0)
print("Appuyez sur 'q' pour quitter l'application.")

# --- 3. BOUCLE PRINCIPALE ---
with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionnement et création du panneau latéral
        frame = cv2.resize(frame, (640, 480))
        h, w, _ = frame.shape
        panel = np.zeros((480, 250, 3), dtype=np.uint8)
        cv2.putText(panel, "Signe detecte :", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # --- A. ANALYSE MEDIAPIPE (Le Squelette) ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # MediaPipe a besoin d'un chronomètre précis
        timestamp_ms = int(time.time() * 1000) 
        mp_results = landmarker.detect_for_video(mp_image, timestamp_ms)

        # --- B. ANALYSE YOLO (La Traduction) ---
        yolo_results = model(frame, stream=True, verbose=False, conf=0.25)

        signe_detecte = False

        # --- C. DESSIN ET FUSION ---
        # 1. On dessine d'abord le squelette s'il est là (Couche du bas)
        if mp_results.hand_landmarks:
            for hand_landmarks in mp_results.hand_landmarks:
                points_pixels = []
                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    points_pixels.append((x, y))
                    # Points orange pour les articulations
                    cv2.circle(frame, (x, y), 5, (0, 150, 255), -1) 

                # Lignes blanches pour les os
                for connexion in connexions:
                    pt1 = points_pixels[connexion[0]]
                    pt2 = points_pixels[connexion[1]]
                    cv2.line(frame, pt1, pt2, (255, 255, 255), 2) 

        # 2. On superpose les infos de YOLO (Couche du haut)
        for r in yolo_results:
            boxes = r.boxes
            if len(boxes) > 0:
                signe_detecte = True
                
                box = boxes[0] 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = 0.25 # float(box.conf[0])
                cls_id = int(box.cls[0])
                lettre_brute = model.names[cls_id]

                # Stabilisateur de texte
                historique_lettres.append(lettre_brute)
                try:
                    lettre_stable = mode(historique_lettres)
                except StatisticsError:
                    lettre_stable = lettre_brute

                # Dessin du cadre vert de ciblage
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
                cv2.putText(frame, lettre_stable, (x2 + 10, max(30, y1 + 30)), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 100), 2)

                # Mise à jour du panneau
                cv2.putText(panel, f"1. {lettre_stable} ({conf:.0%})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

        if not signe_detecte:
            historique_lettres.clear()
            lettre_stable = "..."
            cv2.putText(panel, "Aucune main visible", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        # Assemblage de l'image finale
        fenetre_finale = cv2.hconcat([frame, panel])
        cv2.imshow("Traducteur Langue des Signes PRO", fenetre_finale)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()