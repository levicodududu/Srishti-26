import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- STEP 1: Initialization (Replaces mp_hands.Hands) ---
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

import math

# Steering State
steering_angle = 0.0
calibration_offset = 0.0
is_calibrated = False

# Smoothing (Exponential Moving Average)
# 0.1 = very smooth but laggy | 0.9 = jerky but instant
smoothing_factor = 0.2
smoothed_angle = 0.0


def get_hand_centroid(hand_landmarks):
    """Calculates the average X and Y coordinates for all 21 points of a hand."""
    x_sum = sum([landmark.x for landmark in hand_landmarks])
    y_sum = sum([landmark.y for landmark in hand_landmarks])
    count = len(hand_landmarks)  # This will be 21

    return (x_sum / count), (y_sum / count)


def calculate_distance(point1, point2):
    """A standard math function to find the distance between two X,Y points."""
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

def calculate_3d_distance(point1, point2):
    """A standard math function to find the distance between two X,Y,Z points."""
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2)

def get_thumb_extension(hand_landmarks, min_r, max_r):
    wrist = hand_landmarks[0]
    middle_mcp = hand_landmarks[9]
    pinky_mcp = hand_landmarks[17]
    thumb_tip = hand_landmarks[4]

    # 3D scale is a must for calibration to hold up during turns
    hand_size = calculate_3d_distance(wrist, middle_mcp)
    extension_dist = calculate_distance(pinky_mcp, thumb_tip)

    if hand_size == 0: return 0.0
    
    current_ratio = extension_dist / hand_size
    
    # Use the calibrated min/max instead of hardcoded numbers
    val = (current_ratio - min_r) / (max_r - min_r)
    return max(0.0, min(1.0, val))

# Calibration States
# Start with safe defaults, but these will be overwritten
calib_min_left, calib_max_left = 0.5, 1.0
calib_min_right, calib_max_right = 0.5, 1.0

# Instruction tracking
calib_step = 0

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
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),  # Thumb
        (5, 6),
        (6, 7),
        (7, 8),  # Index finger
        (9, 10),
        (10, 11),
        (11, 12),  # Middle finger
        (13, 14),
        (14, 15),
        (15, 16),  # Ring finger
        (17, 18),
        (18, 19),
        (19, 20),  # Pinky
        (0, 5),
        (5, 9),
        (9, 13),
        (13, 17),
        (0, 17),  # Palm/Knuckles
    ]
    height, width, _ = image.shape
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
                p1 = (
                    int(start_lm.x * image.shape[1]),
                    int(start_lm.y * image.shape[0]),
                )
                p2 = (int(end_lm.x * image.shape[1]), int(end_lm.y * image.shape[0]))

                # Draw the line (Color: White, Thickness: 2)
                cv2.line(image, p1, p2, (255, 255, 255), 2)

            # B. DRAW THE DOTS ON TOP
            for landmark in hand_landmarks:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    if detection_result.hand_landmarks and len(detection_result.hand_landmarks) == 2:

        left_hand_landmarks = None
        right_hand_landmarks = None

        # 1. Ask MediaPipe which hand is which
        for i in range(2):
            # Grab the text label ("Left" or "Right") for the current hand
            hand_label = detection_result.handedness[i][0].category_name

            if hand_label == "Left":
                right_hand_landmarks = detection_result.hand_landmarks[i]
            elif hand_label == "Right":
                left_hand_landmarks = detection_result.hand_landmarks[i]

        # 2. Only proceed if it successfully found one Left and one Right hand
        if left_hand_landmarks and right_hand_landmarks:

            # Get the centroids using your helper function
            left_x, left_y = get_hand_centroid(left_hand_landmarks)
            right_x, right_y = get_hand_centroid(right_hand_landmarks)

            # 3. Calculate the "Hand Vector" Angle using the centroids
            dy = left_y - right_y
            dx = right_x - left_x
            raw_angle = math.degrees(math.atan2(dy, dx))

            # --- CALIBRATION LOGIC ---
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                # Set IDLE (0% Gas/Brake) - Hold thumbs against your palm
                calib_min_left = calculate_distance(left_hand_landmarks[17], left_hand_landmarks[4]) / calculate_3d_distance(left_hand_landmarks[0], left_hand_landmarks[9])
                calib_min_right = calculate_distance(right_hand_landmarks[17], right_hand_landmarks[4]) / calculate_3d_distance(right_hand_landmarks[0], right_hand_landmarks[9])
                calibration_offset = raw_angle # Also zero out steering
                print(f"Zero set! L_Min: {calib_min_left:.2f}, R_Min: {calib_min_right:.2f}")

            elif key == ord('x'):
                # Set MAX (100% Gas/Brake) - Extend thumbs fully
                calib_max_left = calculate_distance(left_hand_landmarks[17], left_hand_landmarks[4]) / calculate_3d_distance(left_hand_landmarks[0], left_hand_landmarks[9])
                calib_max_right = calculate_distance(right_hand_landmarks[17], right_hand_landmarks[4]) / calculate_3d_distance(right_hand_landmarks[0], right_hand_landmarks[9])
                print(f"Max set! L_Max: {calib_max_left:.2f}, R_Max: {calib_max_right:.2f}")

            # --- CALCULATE OUTPUTS ---
            gas_amount = get_thumb_extension(left_hand_landmarks, calib_min_left, calib_max_left)
            brake_amount = get_thumb_extension(right_hand_landmarks, calib_min_right, calib_max_right)

            # 5. Apply Offset and Smooth
            current_angle = raw_angle - calibration_offset
            smoothed_angle = (current_angle * smoothing_factor) + ( smoothed_angle * (1 - smoothing_factor))

            # print(f"Raw Angle: {raw_angle}, Smoothed Angle: {smoothed_angle}")

            # Print it out to see the values update in real-time!


            # # Optional: Draw the gas/brake values on the screen
            # cv2.putText(
            #     image,
            #     f"Gas: {int(gas_amount * 100)}%",
            #     (width - 200, 50),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     1,
            #     (0, 255, 0),
            #     2,
            # )
            # cv2.putText(
            #     image,
            #     f"Brake: {int(brake_amount * 100)}%",
            #     (50, 50),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     1,
            #     (0, 0, 255),
            #     2,
            # )
            # cv2.putText(
            #     image,
            #     f"Steering: {int(smoothed_angle)} deg",
            #     (width // 2 - 130, 50),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     1,
            #     (255, 255, 0),
            #     2,
            # )

            # Simple deadzone: If thumb is just resting, don't accelerate

            print(f"Steering: {int(smoothed_angle)} | Gas: {int(gas_amount*100)} | Brake: {int(brake_amount*100)}")

            # # Optional: Visualize as a vertical bar on the screen
            # # Gas bar (Green)
            # cv2.rectangle(image, (width - 50, 400), (width - 20, 400 - int(gas_amount * 200)), (0, 255, 0), -1)
            # # Brake bar (Red)
            # cv2.rectangle(image, (20, 400), (50, 400 - int(brake_amount * 200)), (0, 0, 255), -1)

        # # 5. Output to "Car"
        # # Map this to your controls (e.g., -90 to 90 degrees)
        # send_to_vehicle(smoothed_angle)

    # Display the result
    # 1. Get the dimensions of the current frame

    # 2. Calculate the vertical center
    center_y = height // 2

    # 3. Draw a thin horizontal line across the middle
    # (image, start_point, end_point, color_bgr, thickness)
    cv2.line(image, (0, center_y), (width, center_y), (0, 0, 255), 1)

    cv2.imshow("MediaPipe Tasks Hands", cv2.flip(image, 1))

    if cv2.waitKey(5) & 0xFF == 27:
        break

detector.close()  # Important to release resources
cap.release()
cv2.destroyAllWindows()