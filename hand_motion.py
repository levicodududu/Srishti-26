import cv2
import fsds
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pathlib
import math
import time

# =========================================================
# 1. MODEL / CAMERA SETUP
# =========================================================
MODEL_PATH = str(pathlib.Path(__file__).parent / "hand_landmarker.task")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)  
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# =========================================================
# 2. FSDS SIMULATOR CONNECTION
# =========================================================
client = fsds.FSDSClient()
client.confirmConnection()
client.enableApiControl(True)
client.reset()

# =========================================================
# 3. STEERING / CALIBRATION STATE
# =========================================================
steering_angle = 0.0
calibration_offset = 0.0

gas_amount = 0.0
brake_amount = 1.0

smoothed_angle = 0.0
smoothing_factor = 0.2  # 0.1 smooth but laggy | 0.9 fast but jerky

PROTRUSION_THRESHOLD = 0.50


class CalibState:
    WAITING = "WAITING"
    COUNTDOWN = "COUNTDOWN"
    CALIBRATED = "CALIBRATED"


calib_state = CalibState.WAITING
countdown_start = 0.0
COUNTDOWN_SECONDS = 3

calib_curl_left = 0.0
calib_curl_right = 0.0

# =========================================================
# 4. DRAWING HELPERS
# =========================================================
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),      # Thumb
    (5, 6),
    (6, 7),
    (7, 8),      # Index
    (9, 10),
    (10, 11),
    (11, 12),    # Middle
    (13, 14),
    (14, 15),
    (15, 16),    # Ring
    (17, 18),
    (18, 19),
    (19, 20),    # Pinky
    (0, 5),
    (5, 9),
    (9, 13),
    (13, 17),
    (0, 17),     # Palm
]


def draw_calibration_overlay(image, state, countdown_start, countdown_seconds):
    height, width, _ = image.shape
    cx = width // 2

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
        remaining = max(0, countdown_seconds - elapsed)
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

        progress = min(1.0, elapsed / countdown_seconds)
        bar_width = int(progress * width)
        cv2.rectangle(image, (0, height - 10), (bar_width, height), (0, 200, 255), -1)

        text = "Hold your resting grip..."
        color = (0, 200, 255)
        y = height - 20

    elif state == CalibState.CALIBRATED:
        text = "Calibrated!"
        color = (0, 255, 80)
        scale = 0.8

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


# =========================================================
# 5. GEOMETRY HELPERS
# =========================================================
def get_hand_centroid(hand_landmarks):
    x_sum = sum(landmark.x for landmark in hand_landmarks)
    y_sum = sum(landmark.y for landmark in hand_landmarks)
    count = len(hand_landmarks)
    return (x_sum / count), (y_sum / count)


def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def calculate_3d_distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x) ** 2
        + (point1.y - point2.y) ** 2
        + (point1.z - point2.z) ** 2
    )


def get_finger_curl(hand_landmarks):
    """
    Measures average extension of middle, ring, pinky fingers
    as distance from fingertip to its own MCP.
    Larger value = more extended
    Smaller value = more curled
    """
    mid = calculate_3d_distance(hand_landmarks[12], hand_landmarks[9])
    ring = calculate_3d_distance(hand_landmarks[16], hand_landmarks[13])
    pinky = calculate_3d_distance(hand_landmarks[20], hand_landmarks[17])
    avg = (mid + ring + pinky) / 3.0

    hand_size = calculate_3d_distance(hand_landmarks[0], hand_landmarks[9])
    return avg / hand_size if hand_size > 0 else 0.0


# =========================================================
# 6. FSDS CONTROL OUTPUT
# =========================================================
def send_to_vehicle(smoothed_angle, gas_amount, brake_amount):
    # Convert approx steering degrees to FSDS [-1, 1]
    steering = max(-1.0, min(1.0, smoothed_angle / 90.0))
    throttle = max(0.0, min(1.0, gas_amount))
    brake = max(0.0, min(1.0, brake_amount))

    car_controls = fsds.CarControls()
    car_controls.steering = steering * 1.2
    car_controls.throttle = throttle*0.8
    car_controls.brake = brake

    client.setCarControls(car_controls)


# =========================================================
# 7. MAIN LOOP
# =========================================================
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)

    height, width, _ = image.shape

    # -----------------------------------------------------
    # Draw hand skeletons
    # -----------------------------------------------------
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            for start_idx, end_idx in HAND_CONNECTIONS:
                start_lm = hand_landmarks[start_idx]
                end_lm = hand_landmarks[end_idx]

                p1 = (
                    int(start_lm.x * width),
                    int(start_lm.y * height),
                )
                p2 = (
                    int(end_lm.x * width),
                    int(end_lm.y * height),
                )
                cv2.line(image, p1, p2, (255, 255, 255), 2)

            for landmark in hand_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    # Defaults for safety
    gas = 0.0
    brake = 1.0
    steer_to_send = 0.0

    # -----------------------------------------------------
    # Detect both hands and classify left/right
    # -----------------------------------------------------
    if detection_result.hand_landmarks and len(detection_result.hand_landmarks) == 2:
        left_hand_landmarks = None
        right_hand_landmarks = None

        for i in range(2):
            hand_label = detection_result.handedness[i][0].category_name

            # Keeping your original swap logic because selfie preview flips things
            if hand_label == "Left":
                right_hand_landmarks = detection_result.hand_landmarks[i]
            elif hand_label == "Right":
                left_hand_landmarks = detection_result.hand_landmarks[i]

        if left_hand_landmarks is not None and right_hand_landmarks is not None:
            left_x, left_y = get_hand_centroid(left_hand_landmarks)
            right_x, right_y = get_hand_centroid(right_hand_landmarks)

            current_curl_left = get_finger_curl(left_hand_landmarks)
            current_curl_right = get_finger_curl(right_hand_landmarks)

            dy = left_y - right_y
            dx = right_x - left_x
            raw_angle = math.degrees(math.atan2(dy, dx))

            # Auto calibration state machine
            if calib_state == CalibState.WAITING:
                calib_state = CalibState.COUNTDOWN
                countdown_start = time.time()

            elif calib_state == CalibState.COUNTDOWN:
                if time.time() - countdown_start >= COUNTDOWN_SECONDS:
                    calibration_offset = raw_angle
                    calib_curl_left = current_curl_left
                    calib_curl_right = current_curl_right
                    calib_state = CalibState.CALIBRATED
                    print("Auto-calibrated!")

            if calib_state == CalibState.CALIBRATED:
                current_angle = raw_angle - calibration_offset
                smoothed_angle = (
                    current_angle * smoothing_factor
                    + smoothed_angle * (1 - smoothing_factor)
                )

                delta_left = current_curl_left - calib_curl_left
                delta_right = current_curl_right - calib_curl_right

                brake = 1.0 if delta_left > PROTRUSION_THRESHOLD else 0.0
                gas = 1.0 if delta_right > PROTRUSION_THRESHOLD else 0.0
                steer_to_send = smoothed_angle

                print(
                    f"Steer: {int(smoothed_angle):>4} | "
                    f"GAS (R): {gas:.0f} | BRAKE (L): {brake:.0f} | "
                    f"ΔL: {delta_left:.3f}  ΔR: {delta_right:.3f}"
                )
            else:
                steer_to_send = 0.0
                gas = 0.0
                brake = 1.0

            # Draw HUD
            cv2.putText(
                image,
                f"Gas: {int(gas * 100)}%",
                (width - 220, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                image,
                f"Brake: {int(brake * 100)}%",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                image,
                f"Steering: {int(steer_to_send)} deg",
                (width // 2 - 160, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )
        else:
            # # Only one true left/right pair not found
            # if calib_state != CalibState.WAITING:
            #     calib_state = CalibState.WAITING
            #     print("Hands lost — will re-calibrate on next appearance.")
            pass
    else:
        # Lost hands entirely
        if calib_state != CalibState.WAITING:
            calib_state = CalibState.WAITING
            print("Hands lost — will re-calibrate on next appearance.")
            client.reset()

    # -----------------------------------------------------
    # Manual calibration / exit keys
    # -----------------------------------------------------
    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):
        if detection_result.hand_landmarks and len(detection_result.hand_landmarks) == 2:
            left_hand_landmarks = None
            right_hand_landmarks = None

            for i in range(2):
                hand_label = detection_result.handedness[i][0].category_name
                if hand_label == "Left":
                    right_hand_landmarks = detection_result.hand_landmarks[i]
                elif hand_label == "Right":
                    left_hand_landmarks = detection_result.hand_landmarks[i]

            if left_hand_landmarks is not None and right_hand_landmarks is not None:
                left_x, left_y = get_hand_centroid(left_hand_landmarks)
                right_x, right_y = get_hand_centroid(right_hand_landmarks)

                dy = left_y - right_y
                dx = right_x - left_x
                raw_angle = math.degrees(math.atan2(dy, dx))

                calibration_offset = raw_angle
                calib_curl_left = get_finger_curl(left_hand_landmarks)
                calib_curl_right = get_finger_curl(right_hand_landmarks)
                calib_state = CalibState.CALIBRATED
                print("Manual calibration complete!")
                
           
        
    if key == 27:  # ESC
        break

    # -----------------------------------------------------
    # Fail-safe control send
    # -----------------------------------------------------
    send_to_vehicle(steer_to_send, gas, brake)

    # Middle guide line
    center_y = height // 2
    cv2.line(image, (0, center_y), (width, center_y), (0, 0, 255), 1)

    draw_calibration_overlay(image, calib_state, countdown_start, COUNTDOWN_SECONDS)

    cv2.imshow("MediaPipe Tasks Hands", cv2.flip(image, 1))

# =========================================================
# 8. CLEANUP
# =========================================================
send_to_vehicle(0.0, 0.0, 1.0)
client.enableApiControl(False)

detector.close()
cap.release()
cv2.destroyAllWindows()