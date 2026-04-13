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
import time


class CalibState:
    WAITING = "WAITING"
    COUNTDOWN = "COUNTDOWN"
    CALIBRATED = "CALIBRATED"


calib_state = CalibState.WAITING
countdown_start = 0.0
COUNTDOWN_SECONDS = 3  # change to whatever length you want
calib_curl_left = 0.0
calib_curl_right = 0.0
# Normalized by hand size. Tune between 0.15–0.30.
# Higher = needs a more deliberate curl inward to trigger.
PROTRUSION_THRESHOLD = 0.50

# Smoothing (Exponential Moving Average)
# 0.1 = very smooth but laggy | 0.9 = jerky but instant
smoothing_factor = 0.2
smoothed_angle = 0.0


def draw_calibration_overlay(image, state, countdown_start, countdown_seconds):
    """Draws the calibration status and countdown on the frame."""
    height, width, _ = image.shape
    cx = width // 2

    # Safe defaults — overwritten by whichever branch matches
    text = ""
    color = (255, 255, 255)
    scale = 0.7
    thickness = 2
    y = height - 30

    if state == CalibState.WAITING:
        text = "Show both hands to calibrate"
        color = (0, 200, 255)

    elif state == CalibState.COUNTDOWN:
        elapsed = time.time() - countdown_start
        remaining = countdown_seconds - elapsed
        number = math.ceil(remaining)

        big_text = str(number)
        big_scale = 5.0
        big_thick = 8

        (tw, th), _ = cv2.getTextSize(
            big_text, cv2.FONT_HERSHEY_SIMPLEX, big_scale, big_thick
        )
        cv2.putText(
            image,
            big_text,
            (cx - tw // 2, height // 2 + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            big_scale,
            (0, 200, 255),
            big_thick,
        )

        bar_width = int((elapsed / countdown_seconds) * width)
        cv2.rectangle(image, (0, height - 10), (bar_width, height), (0, 200, 255), -1)

        text = "Hold your resting grip..."
        color = (0, 200, 255)
        y = height - 20

    elif state == CalibState.CALIBRATED:
        text = "Calibrated!"
        color = (0, 255, 80)
        scale = 0.8

    # Only draw the bottom text if there's something to show
    if text:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.putText(
            image,
            text,
            (cx - tw // 2, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
        )


def get_hand_centroid(hand_landmarks):
    """Calculates the average X and Y coordinates for all 21 points of a hand."""
    x_sum = sum([landmark.x for landmark in hand_landmarks])
    y_sum = sum([landmark.y for landmark in hand_landmarks])
    count = len(hand_landmarks)  # This will be 21

    return (x_sum / count), (y_sum / count)


def calculate_distance(point1, point2):
    """A standard math function to find the 2D distance between two points."""
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def calculate_3d_distance(point1, point2):
    """Calculates the 3D distance between two points."""
    return math.sqrt(
        (point1.x - point2.x) ** 2
        + (point1.y - point2.y) ** 2
        + (point1.z - point2.z) ** 2
    )



def get_finger_curl(hand_landmarks):
    """
    Measures average extension of middle, ring, pinky fingers
    as distance from fingertip to its own base knuckle (MCP).
    Extended = large value. Curled toward palm = small value.
    Completely invariant to hand rotation or steering angle.
    MCP landmarks: 9 (middle), 13 (ring), 17 (pinky)
    Tip landmarks: 12 (middle), 16 (ring), 20 (pinky)
    """
    mid = calculate_3d_distance(hand_landmarks[12], hand_landmarks[9])
    ring = calculate_3d_distance(hand_landmarks[16], hand_landmarks[13])
    pinky = calculate_3d_distance(hand_landmarks[20], hand_landmarks[17])
    avg = (mid + ring + pinky) / 3.0

    hand_size = calculate_3d_distance(hand_landmarks[0], hand_landmarks[9])
    return avg / hand_size if hand_size > 0 else 0.0


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
        (0, 17),  # Palm
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

            wheel_center = ((left_x + right_x) / 2, (left_y + right_y) / 2)

            current_curl_left = get_finger_curl(left_hand_landmarks)
            current_curl_right = get_finger_curl(right_hand_landmarks)
            # 3. Calculate the "Hand Vector" Angle using the centroids
            dy = left_y - right_y
            dx = right_x - left_x
            raw_angle = math.degrees(math.atan2(dy, dx))

            if calib_state == CalibState.WAITING:
                # Both hands just appeared — kick off the countdown
                calib_state = CalibState.COUNTDOWN
                countdown_start = time.time()
            elif calib_state == CalibState.COUNTDOWN:
                if time.time() - countdown_start >= COUNTDOWN_SECONDS:
                    calibration_offset = raw_angle
                    calib_curl_left = current_curl_left
                    calib_curl_right = current_curl_right
                    calib_state = CalibState.CALIBRATED
                    print("Auto-calibrated!")

            # 4. Calibration (Press 'C' while holding your normal F1 Grip)
            if cv2.waitKey(1) & 0xFF == ord("c"):
                calibration_offset = raw_angle
                calib_protrusion_left = current_curl_left
                calib_protrusion_right = current_curl_right
                is_calibrated = True
                print("Calibrated! Steering and Paddles set.")

            # 5. Apply Offset and Smooth
            current_angle = raw_angle - calibration_offset
            smoothed_angle = (current_angle * smoothing_factor) + (
                smoothed_angle * (1 - smoothing_factor)
            )

            # --- 5. F1 PADDLE SHIFTER LOGIC (Auto-Scaling) ---
            # Get current 2D hand size (Wrist to Middle Knuckle) as our "ruler"
            left_hand_size = calculate_distance(
                left_hand_landmarks[0], left_hand_landmarks[9]
            )
            right_hand_size = calculate_distance(
                right_hand_landmarks[0], right_hand_landmarks[9]
            )

            # Trigger is 1 if the delta passes our negative threshold
            brake = gas = 0
            if calib_state == CalibState.CALIBRATED:
                current_angle = raw_angle - calibration_offset
                smoothed_angle = (current_angle * smoothing_factor) + (
                    smoothed_angle * (1 - smoothing_factor)
                )

                delta_left = current_curl_left - calib_curl_left
                delta_right = current_curl_right - calib_curl_right

                brake = 1 if delta_left > PROTRUSION_THRESHOLD else 0
                gas = 1 if delta_right > PROTRUSION_THRESHOLD else 0

                print(
                    f"Steer: {int(smoothed_angle):>4} | GAS (R): {gas} | BRAKE (L): {brake} | ΔL: {delta_left:.3f}  ΔR: {delta_right:.3f}"
                )

        # # 5. Output to "Car"
        # # Map this to your controls (e.g., -90 to 90 degrees)
        # send_to_vehicle(smoothed_angle)
    else:
        # Lost both hands — drop back to WAITING so it re-calibrates on return
        if calib_state != CalibState.WAITING:
            calib_state = CalibState.WAITING
            print("Hands lost — will re-calibrate on next appearance.")

    # Display the result
    # 1. Get the dimensions of the current frame
    height, width, _ = image.shape

    # 2. Calculate the vertical center
    center_y = height // 2

    # 3. Draw a thin horizontal line across the middle
    # (image, start_point, end_point, color_bgr, thickness)
    cv2.line(image, (0, center_y), (width, center_y), (0, 0, 255), 1)
    draw_calibration_overlay(image, calib_state, countdown_start, COUNTDOWN_SECONDS)

    cv2.imshow("MediaPipe Tasks Hands", cv2.flip(image, 1))

    if cv2.waitKey(5) & 0xFF == 27:
        break

detector.close()  # Important to release resources
cap.release()
cv2.destroyAllWindows()
