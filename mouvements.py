import cv2
import numpy as np
from ultralytics import YOLO

# On utilise le modèle "pose" pour repérer les poignets
model = YOLO('yolov8n-pose.pt')

cap = cv2.VideoCapture(0)

print("Appuyez sur 'q' pour quitter.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # On fixe la taille de la vidéo
    frame = cv2.resize(frame, (640, 480))
    
    # Création du panneau latéral noir pour les probabilités
    panel = np.zeros((480, 250, 3), dtype=np.uint8)
    cv2.putText(panel, "Signe detecte :", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # YOLO analyse l'image
    results = model(frame, stream=True, verbose=False)

    for r in results:
        # Si un corps est détecté
        if r.keypoints is not None and len(r.keypoints) > 0:
            
            # On parcourt toutes les personnes à l'écran
            for i in range(len(r.keypoints)):
                # Coordonnées des 17 points d'articulation
                keypoints = r.keypoints.xy[i]
                # Indice de confiance de chaque point
                confs = r.keypoints.conf[i] if r.keypoints.conf is not None else [1]*17
                
                # Dans YOLO Pose : 9 = poignet gauche, 10 = poignet droit
                poignets = [(9, "Main G"), (10, "Main D")]

                for index, label in poignets:
                    # Si le poignet est visible à l'écran (confiance > 50%)
                    if len(keypoints) > index and confs[index] > 0.5:
                        px, py = int(keypoints[index][0]), int(keypoints[index][1])
                        
                        # On crée une "boîte" de 120x120 pixels centrée sur le poignet pour englober la main
                        taille = 60 
                        x1, y1 = max(0, px - taille), max(0, py - taille)
                        x2, y2 = min(640, px + taille), min(480, py + taille)

                        # 1. Dessiner le cadre UNIQUEMENT autour de la main
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
                        
                        # 2. Écrire la lettre (Simulation avec "A" pour le moment)
                        lettre = "A" 
                        cv2.putText(frame, lettre, (x2 + 10, y1 + 30), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 100), 2)

                        # 3. Mettre à jour le panneau latéral
                        cv2.putText(panel, f"1. {lettre} (85%)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
                        cv2.putText(panel, f"2. B (10%)", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                        cv2.putText(panel, f"3. C (5%)", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    # Assembler l'image de la caméra et le panneau latéral
    fenetre_finale = cv2.hconcat([frame, panel])
    cv2.imshow("Traducteur Langue des Signes", fenetre_finale)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()