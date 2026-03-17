import cv2
import mediapipe as mp
import time

# --- 1. Configuration de la nouvelle API ---
model_path = 'hand_landmarker.task' # Le fichier que vous venez de télécharger

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Configuration pour le flux vidéo (webcam)
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2)

# --- 2. Notre dictionnaire d'os pour dessiner la main ---
# Chaque chiffre est une articulation (0 = poignet, 4 = bout du pouce, 8 = bout de l'index...)
connexions = [
    (0,1), (1,2), (2,3), (3,4),       # Pouce
    (0,5), (5,6), (6,7), (7,8),       # Index
    (5,9), (9,10), (10,11), (11,12),  # Majeur
    (9,13), (13,14), (14,15), (15,16),# Annulaire
    (13,17), (0,17), (17,18), (18,19), (19,20) # Auriculaire
]

print("Appuyez sur 'q' pour quitter.")
cap = cv2.VideoCapture(0)

# Horloge artificielle pour MediaPipe
timestamp_ms = 0

# --- 3. Lancement de la reconnaissance ---
with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe exige son propre format d'image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Le mode vidéo exige un temps qui augmente toujours
        timestamp_ms += 1 

        # Détection !
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        # --- 4. Dessin manuel du squelette ---
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                h, w, _ = frame.shape
                
                # Étape A : Convertir les coordonnées relatives en vrais pixels
                points_pixels = []
                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    points_pixels.append((x, y))
                    
                    # On dessine un point rouge pour chaque articulation
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

                # Étape B : Relier les points pour faire les os (lignes vertes)
                for connexion in connexions:
                    pt1 = points_pixels[connexion[0]]
                    pt2 = points_pixels[connexion[1]]
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        cv2.imshow('Nouvelle API MediaPipe - Articulations', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()