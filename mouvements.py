import cv2
from ultralytics import YOLO

# Chargement du modèle de posture (la version "nano" pour que ce soit rapide)
# Le fichier 'yolov8n-pose.pt' se téléchargera automatiquement au premier lancement
model = YOLO('yolov8n-pose.pt')

# Ouverture de la webcam
cap = cv2.VideoCapture(0)

print("Appuyez sur 'q' pour quitter l'application.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erreur: Impossible de lire la caméra.")
        break

    # YOLOv8 analyse l'image, détecte la posture et garde les résultats en mémoire
    # stream=True optimise le flux pour la vidéo en temps réel
    results = model(frame, stream=True)
    
    # On parcourt les résultats (il y en a un par image)
    for r in results:
        # La fonction plot() dessine automatiquement le squelette sur l'image !
        annotated_frame = r.plot()

    # Affichage de l'image
    cv2.imshow("Reconnaissance des mouvements - YOLOv8", annotated_frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libération des ressources
cap.release()
cv2.destroyAllWindows()