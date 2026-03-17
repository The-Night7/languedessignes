import cv2
import numpy as np
import time
from collections import deque
from statistics import mode, StatisticsError
from ultralytics import YOLO
import mediapipe as mp

# --- 1. CHARGEMENT DES IA ---
print("Chargement du Systeme Hybride...")
model_yolo = YOLO('best.pt')

model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1)

# --- 2. OUTILS DE STABILISATION (NOUVEAU) ---
historique_lettres = deque(maxlen=7) # Filtre anti-clignotement (7 images)
lettre_stable = "..."
confiance_stable = 0.0

# Variables pour le lissage du cadre (anti-tremblement)
# 0.6 signifie qu'on garde 60% de l'ancienne position pour une transition fluide
lissage = 0.6 
old_x1, old_y1, old_x2, old_y2 = 0, 0, 0, 0

connexions = [
    (0,1), (1,2), (2,3), (3,4),       
    (0,5), (5,6), (6,7), (7,8),       
    (5,9), (9,10), (10,11), (11,12),  
    (9,13), (13,14), (14,15), (15,16),
    (13,17), (0,17), (17,18), (18,19), (19,20)
]

cap = cv2.VideoCapture(0)
print("Appuyez sur 'q' pour quitter.")

# --- 3. BOUCLE DE TRAITEMENT OPTIMISÉE ---
with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

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

            points_pixels = []
            x_coords, y_coords = [], []
            
            for landmark in hand_landmarks:
                x, y = int(landmark.x * w), int(landmark.y * h)
                points_pixels.append((x, y))
                x_coords.append(x)
                y_coords.append(y)

            # --- AMÉLIORATION : RECADRAGE CARRÉ DYNAMIQUE ---
            # On force un crop 100% carré pour ne pas écraser les doigts de YOLO
            largeur_main = max(x_coords) - min(x_coords)
            hauteur_main = max(y_coords) - min(y_coords)
            taille_carre = max(largeur_main, hauteur_main)
            
            # Marge adaptative (25% de la taille de la main)
            marge = int(taille_carre * 0.25)
            taille_carre += marge * 2

            # Calcul du centre exact de la main
            cx = (min(x_coords) + max(x_coords)) // 2
            cy = (min(y_coords) + max(y_coords)) // 2

            x1 = max(0, cx - taille_carre // 2)
            y1 = max(0, cy - taille_carre // 2)
            x2 = min(w, cx + taille_carre // 2)
            y2 = min(h, cy + taille_carre // 2)

            # --- AMÉLIORATION : LISSAGE SPATIAL ANTI-TREMBLEMENT ---
            if old_x2 == 0: # Si c'est la première image
                old_x1, old_y1, old_x2, old_y2 = x1, y1, x2, y2
            else:
                # Moyenne mathématique entre l'image précédente et l'actuelle
                x1 = int(old_x1 * lissage + x1 * (1 - lissage))
                y1 = int(old_y1 * lissage + y1 * (1 - lissage))
                x2 = int(old_x2 * lissage + x2 * (1 - lissage))
                y2 = int(old_y2 * lissage + y2 * (1 - lissage))
            
            # On sauvegarde pour la prochaine image
            old_x1, old_y1, old_x2, old_y2 = x1, y1, x2, y2

            # Découpage ultra-propre et stable
            image_isolee = frame[y1:y2, x1:x2]

            # --- B. YOLO (Traduction sur image propre) ---
            if image_isolee.size != 0:
                # Utilisation de .predict (légèrement plus optimisé que l'appel direct)
                yolo_results = model_yolo.predict(image_isolee, verbose=False, conf=0.30)
                
                meilleure_conf = 0
                lettre_brute = None

                for r in yolo_results:
                    if len(r.boxes) > 0:
                        box = r.boxes[0] 
                        conf = float(box.conf[0])
                        # Si l'IA trouve plusieurs choses, on force la meilleure note
                        if conf > meilleure_conf:
                            meilleure_conf = conf
                            lettre_brute = model_yolo.names[int(box.cls[0])]

                # --- AMÉLIORATION : FILTRE TEMPOREL ---
                if lettre_brute is not None:
                    historique_lettres.append(lettre_brute)
                    try:
                        # On prend la lettre la plus vue sur les 7 dernières images
                        lettre_stable = mode(historique_lettres)
                        confiance_stable = meilleure_conf
                    except StatisticsError:
                        pass # En cas d'égalité stricte, on garde la lettre précédente

            # --- C. INTERFACE VISUELLE ---
            # Squelette MediaPipe
            for connexion in connexions:
                cv2.line(frame, points_pixels[connexion[0]], points_pixels[connexion[1]], (255, 255, 255), 2)
            for pt in points_pixels:
                cv2.circle(frame, pt, 4, (0, 150, 255), -1)

            # Cadre YOLO
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
            cv2.putText(frame, lettre_stable, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 100), 2)

            # Panneau
            cv2.putText(panel, f"Signe: {lettre_stable}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
            cv2.putText(panel, f"Preci: {confiance_stable:.0%}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        else:
            # Réinitialisation si la main disparaît
            historique_lettres.clear()
            old_x2 = 0 
            lettre_stable = "..."
            cv2.putText(panel, "Aucune main", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        fenetre_finale = cv2.hconcat([frame, panel])
        cv2.imshow("Pipeline Langue des Signes PRO", fenetre_finale)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()