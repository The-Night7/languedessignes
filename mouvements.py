import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from statistics import mode, StatisticsError

# 1. Chargement de votre modèle entraîné
model = YOLO('modele_signes.pt')

cap = cv2.VideoCapture(0)

# Mémoire des 5 dernières prédictions pour éviter les clignotements
historique_lettres = deque(maxlen=5)
lettre_stable = "..."

print("Appuyez sur 'q' pour quitter.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    panel = np.zeros((480, 250, 3), dtype=np.uint8)
    cv2.putText(panel, "Signe detecte :", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 2. SEUIL DE CONFIANCE : on ajoute conf=0.65 (65% minimum)
    results = model(frame, stream=True, verbose=False, conf=0.65)

    signe_detecte = False

    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            signe_detecte = True
            
            # On prend la détection la plus certaine
            box = boxes[0] 
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            lettre_brute = model.names[cls_id]

            # 3. STABILISATEUR : On ajoute la lettre à l'historique
            historique_lettres.append(lettre_brute)
            
            # On cherche la lettre la plus fréquente dans les 5 dernières images
            try:
                lettre_stable = mode(historique_lettres)
            except StatisticsError:
                lettre_stable = lettre_brute # En cas d'égalité, on garde la dernière

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
            cv2.putText(frame, lettre_stable, (x2 + 10, max(30, y1 + 30)), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 100), 2)

            cv2.putText(panel, f"1. {lettre_stable} ({conf:.0%})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

    if not signe_detecte:
        # Si on ne voit plus de main, on vide la mémoire
        historique_lettres.clear()
        lettre_stable = "..."
        cv2.putText(panel, "Aucune main visible", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    fenetre_finale = cv2.hconcat([frame, panel])
    cv2.imshow("Traducteur Langue des Signes", fenetre_finale)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()