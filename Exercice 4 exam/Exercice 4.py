import cv2

# Choisissez l'algorithme de suivi : KCF ou CSRT
tracker = cv2.TrackerCSRT_create()  # Pour utiliser KCF, remplacez par cv2.TrackerKCF_create()

# Ouvrir la vidéo de la webcam
cap = cv2.VideoCapture(0)

# Lire la première image
ret, frame = cap.read()

# Sélectionner la région d'intérêt (ROI) pour l'objet à suivre
bbox = cv2.selectROI("Sélectionner l'objet à suivre", frame, fromCenter=False, showCrosshair=True)

# Initialiser le tracker avec la première image et le rectangle
tracker.init(frame, bbox)

# Boucle de suivi
while True:
    # Lire la nouvelle image
    ret, frame = cap.read()
    if not ret:
        break

    # Mettre à jour le tracker
    success, bbox = tracker.update(frame)

    # Vérifier si le suivi a réussi
    if success:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Dessiner le rectangle
    else:
        cv2.putText(frame, "Objet perdu", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Afficher message

    # Afficher l'image avec le suivi
    cv2.imshow("Suivi d'objet en temps réel", frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la caméra et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
