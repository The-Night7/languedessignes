import cv2
import numpy as np
from ultralytics import YOLO

# Chargement du modèle. 
# REMARQUE : Pour l'instant on utilise le modèle de base. 
# Plus tard, vous remplacerez 'yolov8n.pt' par 'votre_modele_langue_des_signes.pt'
model = YOLO('yolov8n.pt')

# Ouverture de la webcam
cap = cv2.VideoCapture(0)

print("Appuyez sur 'q' pour quitter l'application.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # On redimensionne la caméra pour avoir une taille fixe et propre (ex: 640x480)
    frame = cv2.resize(frame, (640, 480))
    
    # Création du panneau latéral noir (largeur: 250px, hauteur: 480px, 3 canaux de couleur)
    panel = np.zeros((480, 250, 3), dtype=np.uint8)

    # YOLO analyse l'image
    results = model(frame, stream=True, verbose=False)

    # Titre dans le panneau latéral
    cv2.putText(panel, "Signe detecte :", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Récupération des coordonnées de la boîte englobante (la main)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Récupération de la confiance (probabilité) et de la classe (la lettre)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            lettre = model.names[cls_id] # C'est ici que ça affichera "A", "B", etc. avec le bon modèle

            # 1. Dessiner un rectangle autour de la main
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
            
            # 2. Écrire la lettre juste à côté de la main
            cv2.putText(frame, lettre, (x2 + 10, y1 + 30), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 100), 2)

            # 3. Afficher les signes les plus probables dans le panneau latéral
            # Note : YOLO standard renvoie la meilleure détection. Pour simuler l'interface, 
            # on affiche la meilleure, et on prépare visuellement la place pour les suivantes.
            cv2.putText(panel, f"1. {lettre} ({conf:.0%})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
            cv2.putText(panel, f"2. Autre ({(conf*0.7):.0%})", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            cv2.putText(panel, f"3. Autre ({(conf*0.4):.0%})", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    # Concaténer (coller) la caméra et le panneau latéral côte à côte
    fenetre_finale = cv2.hconcat([frame, panel])

    # Affichage
    cv2.imshow("Traducteur Langue des Signes", fenetre_finale)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()