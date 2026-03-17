import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from statistics import mode, StatisticsError

model = YOLO('best.pt')
cap = cv2.VideoCapture(0)

# Mémoire des prédictions (pour la stabilité)
historique_lettres = deque(maxlen=5)
lettre_stable = "..."

# NOUVEAU : Mémoire des coordonnées de la main (pour dessiner le mouvement)
trajectoire = deque(maxlen=20)

print("Appuyez sur 'q' pour quitter.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    panel = np.zeros((480, 250, 3), dtype=np.uint8)
    cv2.putText(panel, "Signe detecte :", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # NOUVEAU : On utilise model.track() au lieu de model()
    # persist=True dit à l'IA de mémoriser la main d'une image à l'autre
    results = model.track(frame, persist=True, conf=0.60, verbose=False)

    signe_detecte = False

    # On vérifie s'il y a des détections ET si elles ont un ID de suivi
    if results[0].boxes is not None and results[0].boxes.id is not None:
        signe_detecte = True
        
        boxes = results[0].boxes
        box = boxes[0] 
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        lettre_brute = model.names[cls_id]

        # Stabilisateur de texte
        historique_lettres.append(lettre_brute)
        try:
            lettre_stable = mode(historique_lettres)
        except StatisticsError:
            lettre_stable = lettre_brute

        # NOUVEAU : Calcul du centre de la main pour le mouvement
        centre_x = int((x1 + x2) / 2)
        centre_y = int((y1 + y2) / 2)
        trajectoire.append((centre_x, centre_y))

        # NOUVEAU : Dessin de la "traînée" du mouvement (une ligne qui suit la main)
        for i in range(1, len(trajectoire)):
            # L'épaisseur de la ligne diminue pour faire un effet de "queue de comète"
            epaisseur = int(np.sqrt(20 / float(i + 1)) * 3)
            cv2.line(frame, trajectoire[i - 1], trajectoire[i], (255, 0, 255), epaisseur)

        # Affichage du cadre et du texte
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
        cv2.putText(frame, lettre_stable, (x2 + 10, max(30, y1 + 30)), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 100), 2)

        cv2.putText(panel, f"1. {lettre_stable} ({conf:.0%})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

    if not signe_detecte:
        # Si on perd la main, on vide les mémoires progressivement
        historique_lettres.clear()
        trajectoire.clear()
        lettre_stable = "..."
        cv2.putText(panel, "Aucune main visible", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    fenetre_finale = cv2.hconcat([frame, panel])
    cv2.imshow("Traducteur Langue des Signes", fenetre_finale)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()