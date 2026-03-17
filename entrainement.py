from ultralytics import YOLO

# 1. On charge un modèle "vide" mais pré-structuré (YOLOv8 nano, très rapide)
model = YOLO('yolov8n.pt')

if __name__ == '__main__':
    print("Début de l'entraînement de l'IA...")
    
    # 2. On lance l'apprentissage
    # Assurez-vous que le chemin vers data.yaml est correct
    results = model.train(
        data='dataset/data.yaml', # Le plan de cours
        epochs=15,                # Le nombre de fois que l'IA va lire tout le dataset (15 est un bon début pour tester)
        imgsz=640,                # La taille des images
        plots=True                # Pour générer des graphiques de progression
    )
    
    print("Entraînement terminé !")