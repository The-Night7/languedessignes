import cv2
import numpy as np
import time
from ultralytics import YOLO
import mediapipe as mp

# --- 1. CHARGEMENT DES DEUX IA ---
print("Chargement des cerveaux virtuels...")
model_yolo = YOLO('best.pt')

model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1) # On se concentre sur une seule main pour l'instant

# --- 2. CONFIGURATION VISUELLE ---
connexions = [
    (0,1), (1,2), (2,3), (3,4),       # Pouce
    (0,5), (5,6), (6,7), (7,8),       # Index
    (5,9), (9,10), (10,11), (11,12),  # Majeur
    (9,13), (13,14), (14,15), (15,16),# Annulaire
    (13,17), (0,17), (17,18), (18,19), (19,20) # Auriculaire
]

cap = cv2.VideoCapture(0)
print("Appuyez sur 'q' pour quitter l'application.")

# --- 3. LA BOUCLE DE TRAVAIL D'ÉQUIPE ---
with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        h, w, _ = frame.shape
        
        # Création du panneau latéral
        panel = np.zeros((480, 250, 3), dtype=np.uint8)
        cv2.putText(panel, "SYSTEME HYBRIDE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.line(panel, (10, 50), (240, 50), (255, 255, 255), 1)

        # --- ÉTAPE A : MEDIAPIPE CHERCHE LA MAIN ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000) 
        mp_results = landmarker.detect_for_video(mp_image, timestamp_ms)

        lettre_predite = "..."
        confiance = 0.0

        if mp_results.hand_landmarks:
            hand_landmarks = mp_results.hand_landmarks[0]

            # 1. On récupère toutes les coordonnées de la main
            points_pixels = []
            x_coords = []
            y_coords = []
            
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points_pixels.append((x, y))
                x_coords.append(x)
                y_coords.append(y)

            # 2. On calcule les limites de la main pour créer un recadrage (Crop)
            marge = 40 # On ajoute 40 pixels de marge pour ne pas couper le bout des doigts
            x1 = max(0, min(x_coords) - marge)
            y1 = max(0, min(y_coords) - marge)
            x2 = min(w, max(x_coords) + marge)
            y2 = min(h, max(y_coords) + marge)

            # 3. On "découpe" virtuellement l'image pour l'isoler
            image_isolee = frame[y1:y2, x1:x2]

            # --- ÉTAPE B : YOLO TRADUIT L'IMAGE ISOLÉE ---
            # Si la découpe est valide (pas d'erreur de bordure d'écran)
            if image_isolee.size != 0:
                # YOLO ne regarde plus toute la pièce, il regarde JUSTE la main !
                yolo_results = model_yolo(image_isolee, verbose=False, conf=0.25)
                
                for r in yolo_results:
                    if len(r.boxes) > 0:
                        box = r.boxes[0] 
                        cls_id = int(box.cls[0])
                        confiance = float(box.conf[0])
                        lettre_predite = model_yolo.names[cls_id]

            # --- ÉTAPE C : DESSIN DE L'INTERFACE ---
            # Dessin des articulations (MediaPipe)
            for connexion in connexions:
                pt1 = points_pixels[connexion[0]]
                pt2 = points_pixels[connexion[1]]
                cv2.line(frame, pt1, pt2, (255, 255, 255), 2)
            for pt in points_pixels:
                cv2.circle(frame, pt, 4, (0, 150, 255), -1)

            # Dessin de la cible et de la lettre (MediaPipe donne la position, YOLO donne la lettre)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
            cv2.putText(frame, lettre_predite, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 100), 2)

            # Mise à jour du panneau
            cv2.putText(panel, f"Signe: {lettre_predite}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
            cv2.putText(panel, f"Preci: {confiance:.0%}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        else:
            cv2.putText(panel, "Aucune main", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Affichage
        fenetre_finale = cv2.hconcat([frame, panel])
        cv2.imshow("Pipeline Langue des Signes", fenetre_finale)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Fin de l'application.")