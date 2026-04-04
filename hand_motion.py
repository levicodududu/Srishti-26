import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- STEP 1: Initialization (Replaces mp_hands.Hands) ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # --- STEP 2: Image Conversion ---
    # Tasks API requires a MediaPipe Image object
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # --- STEP 3: Detection (Replaces hands.process) ---
    detection_result = detector.detect(mp_image)

    # --- STEP 4: Drawing (The Manual Way) ---
    # Since mp_drawing is legacy, we draw the 21 points manually

    # The connection map for the hand skeleton
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),             # Thumb
        (5, 6), (6, 7), (7, 8),                     # Index finger
        (9, 10), (10, 11), (11, 12),                # Middle finger
        (13, 14), (14, 15), (15, 16),               # Ring finger
        (17, 18), (18, 19), (19, 20),               # Pinky
        (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)  # Palm/Knuckles
    ]
    # --- STEP 4: Drawing (The Manual Way) ---
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            
            # A. DRAW THE LINES FIRST (so they are behind the dots)
            for connection in HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                
                # Get the start and end points
                start_lm = hand_landmarks[start_idx]
                end_lm = hand_landmarks[end_idx]
                
                # Convert normalized (0-1) to pixel coordinates
                p1 = (int(start_lm.x * image.shape[1]), int(start_lm.y * image.shape[0]))
                p2 = (int(end_lm.x * image.shape[1]), int(end_lm.y * image.shape[0]))
                
                # Draw the line (Color: White, Thickness: 2)
                cv2.line(image, p1, p2, (255, 255, 255), 2)

            # B. DRAW THE DOTS ON TOP
            for landmark in hand_landmarks:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)


    # Display the result
    cv2.imshow('MediaPipe Tasks Hands', cv2.flip(image, 1))
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

detector.close() # Important to release resources
cap.release()
cv2.destroyAllWindows()