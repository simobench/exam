import cv2
import numpy as np

# Charger le classificateur Haar pour la détection des mains (ajoutez le chemin de votre fichier cascade ici)
hand_cascade = cv2.CascadeClassifier('C:\\Users\\pc\\Desktop\\Partie 3 Bonus\\haarcascade_hand.xml')  # Remplacez par votre modèle de détection de mains

# Ouvrir le flux vidéo (caméra)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les mains dans l'image
    hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Pour chaque main détectée
    for (x, y, w, h) in hands:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Dessiner un rectangle autour de la main

        # Extraire la région de la main pour analyse des gestes
        hand_region = gray[y:y+h, x:x+w]

        # Traitement d'image pour détecter si la main est ouverte ou fermée
        # (ici, vous pourriez ajouter une logique plus complexe pour différencier les gestes)
        hand_threshold = cv2.threshold(hand_region, 50, 255, cv2.THRESH_BINARY)[1]  # Seuillage

        # Compter les contours pour détecter les gestes
        contours, _ = cv2.findContours(hand_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Déterminer le geste basé sur les contours
        if len(contours) > 0:
            # Vous pourriez faire des analyses plus complexes ici
            # Pour cet exemple, nous allons juste compter le nombre de contours
            if len(contours) == 1:  # Supposons qu'un seul contour indique un poing
                gesture_text = "Poing fermé"
            else:  # Plus de contours indiquent une main ouverte
                gesture_text = "Main ouverte"
        else:
            gesture_text = "Aucun geste détecté"

        # Afficher le geste détecté
        cv2.putText(frame, gesture_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Afficher le flux vidéo avec la détection et les gestes
    cv2.imshow("Détection de mains et reconnaissance de gestes", frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la caméra et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
