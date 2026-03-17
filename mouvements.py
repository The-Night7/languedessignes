import cv2
import numpy as np
from ultralytics import YOLO

# Chargement d'un modèle spécifiquement entraîné pour la langue des signes
# (Voir les instructions en dessous pour obtenir ce fichier)
try:
    model = YOLO('modele_signes.pt')
except Exception as e:
    print("Erreur : Le fichier 'modele_signes.pt' est introuvable dans le dossier.")
    print("Veuillez suivre l'étape 2 pour le télécharger.")
    exit()

cap = cv2.VideoCapture(0)

print("Appuyez sur 'q' pour quitter.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    
    # Création du panneau latéral
    panel = np.zeros((480, 250, 3), dtype=np.uint8)
    cv2.putText(panel, "Signe detecte :", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # YOLO détecte les mains/doigts et lit le signe
    results = model(frame, stream=True, verbose=False)

    signe_detecte = False

    for r in results:
        boxes = r.boxes
        
        # S'il trouve des mains faisant un signe
        if len(boxes) > 0:
            signe_detecte = True
            
            # On prend la main la plus visible
            box = boxes[0] 
            
            # Coordonnées qui encadrent EXACTEMENT la main et les doigts
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Récupération de la lettre prédite par le modèle
            lettre = model.names[cls_id]

            # 1. Dessiner le cadre autour des doigts et de la main
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
            
            # 2. Écrire la lettre à côté de la main
            cv2.putText(frame, lettre, (x2 + 10, max(30, y1 + 30)), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 100), 2)

            # 3. Mettre à jour le panneau latéral
            cv2.putText(panel, f"1. {lettre} ({conf:.0%})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
            
            # (Optionnel) Si le modèle hésite avec d'autres signes, on pourrait les lister ici
            cv2.putText(panel, "En attente...", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    if not signe_detecte:
        cv2.putText(panel, "Aucune main visible", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    # Concaténer l'image et le panneau
    fenetre_finale = cv2.hconcat([frame, panel])
    cv2.imshow("Traducteur Langue des Signes", fenetre_finale)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()