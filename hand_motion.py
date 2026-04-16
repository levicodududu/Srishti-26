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
    min_hand_detection_confidence=0.35,
    min_tracking_confidence=0.35,
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# =========================================================
# 2. FSDS SIMULATOR CONNECTION
# =========================================================
client = fsds.FSDSClient()
client.confirmConnection()
client.enableApiControl(True)
client.reset()

# =========================================================
# 3. CONTROL / CALIBRATION STATE
# =========================================================
steering_angle = 0.0
calibration_offset = 0.0

smoothed_angle = 0.0
steering_smoothing_factor = 0.20   # 0.1 smoother, 0.3 quicker

# left open / right open detection
PROTRUSION_THRESHOLD = 0.50

hands_lost_time = None
LOST_THRESHOLD_SECONDS = 5.0

# behavior:
# left hand open  -> activate speed hold to target speed
# right hand open -> hand brake
TARGET_SPEED_MPS = 4.0

# command smoothing
smoothed_steering_cmd = 0.0
smoothed_drive_cmd = 0.0  # + => throttle, - => brake
STEER_ALPHA = 0.20
DRIVE_ALPHA = 0.15

# PID for speed hold
class PIDController:
    def __init__(self, kp, ki, kd, integral_limit=2.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = None
        self.integral_limit = integral_limit

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def compute(self, current, target, dt):
        error = target - current
        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / max(dt, 1e-3)
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

speed_pid = PIDController(kp=0.8, ki=0.15, kd=0.08)

class CalibState:
    WAITING = "WAITING"
    COUNTDOWN = "COUNTDOWN"
    CALIBRATED = "CALIBRATED"

calib_state = CalibState.WAITING
countdown_start = 0.0
COUNTDOWN_SECONDS = 3

calib_curl_left = 0.0
calib_curl_right = 0.0

# performance helpers
frame_count = 0
PROCESS_EVERY_N_FRAMES = 1   # set to 2 if you want less CPU use
last_print_time = 0.0
PRINT_INTERVAL = 0.20
last_loop_time = time.time()

# =========================================================
# 4. DRAWING HELPERS
# =========================================================
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
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

def calculate_3d_distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x) ** 2
        + (point1.y - point2.y) ** 2
        + (point1.z - point2.z) ** 2
    )

def get_finger_curl(hand_landmarks):
    mid = calculate_3d_distance(hand_landmarks[12], hand_landmarks[9])
    ring = calculate_3d_distance(hand_landmarks[16], hand_landmarks[13])
    pinky = calculate_3d_distance(hand_landmarks[20], hand_landmarks[17])
    avg = (mid + ring + pinky) / 3.0

    hand_size = calculate_3d_distance(hand_landmarks[0], hand_landmarks[9])
    return avg / hand_size if hand_size > 0 else 0.0

def get_vehicle_speed():
    state = client.getCarState()
    vx = state.kinematics_estimated.linear_velocity.x_val
    vy = state.kinematics_estimated.linear_velocity.y_val
    return math.hypot(vx, vy)

# =========================================================
# 6. FSDS CONTROL OUTPUT
# =========================================================
def send_to_vehicle(steering_cmd, drive_cmd):
    """
    steering_cmd in [-1, 1]
    drive_cmd in [-1, 1] where:
      +1 = full throttle
      -1 = full brake
    """
    steering_cmd = max(-1.0, min(1.0, steering_cmd))
    drive_cmd = max(-1.0, min(1.0, drive_cmd))

    throttle = max(0.0, drive_cmd)
    brake = max(0.0, -drive_cmd)

    car_controls = fsds.CarControls()
    car_controls.steering = steering_cmd * 1.5  # amplify steering for more responsiveness
    car_controls.throttle = throttle
    car_controls.brake = brake
    client.setCarControls(car_controls)

# =========================================================
# 7. MAIN LOOP
# =========================================================
last_detection_result = None

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    now = time.time()
    dt = max(now - last_loop_time, 1e-3)
    last_loop_time = now

    frame_count += 1

    # Run detection every N frames
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        last_detection_result = detection_result
    else:
        detection_result = last_detection_result

    height, width, _ = image.shape

    left_hand_landmarks = None
    right_hand_landmarks = None

    # -----------------------------------------------------
    # Draw hand skeletons
    # -----------------------------------------------------
    if detection_result and detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            for start_idx, end_idx in HAND_CONNECTIONS:
                start_lm = hand_landmarks[start_idx]
                end_lm = hand_landmarks[end_idx]

                p1 = (int(start_lm.x * width), int(start_lm.y * height))
                p2 = (int(end_lm.x * width), int(end_lm.y * height))
                cv2.line(image, p1, p2, (255, 255, 255), 2)

            for landmark in hand_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    # default safe output
    steer_cmd = 0.0
    drive_cmd = 0.0
    left_open = False
    right_open = False

    # -----------------------------------------------------
    # Detect both hands and classify left/right
    # -----------------------------------------------------
    if detection_result and detection_result.hand_landmarks and len(detection_result.hand_landmarks) == 2:
        for i in range(2):
            hand_label = detection_result.handedness[i][0].category_name

            # keep your selfie-flip mapping
            if hand_label == "Left":
                right_hand_landmarks = detection_result.hand_landmarks[i]
            elif hand_label == "Right":
                left_hand_landmarks = detection_result.hand_landmarks[i]

        if left_hand_landmarks is not None and right_hand_landmarks is not None:
            hands_lost_time = None

            left_x, left_y = get_hand_centroid(left_hand_landmarks)
            right_x, right_y = get_hand_centroid(right_hand_landmarks)

            current_curl_left = get_finger_curl(left_hand_landmarks)
            current_curl_right = get_finger_curl(right_hand_landmarks)

            dy = left_y - right_y
            dx = right_x - left_x
            raw_angle = math.degrees(math.atan2(dy, dx))

            if calib_state == CalibState.WAITING:
                calib_state = CalibState.COUNTDOWN
                countdown_start = time.time()

            elif calib_state == CalibState.COUNTDOWN:
                if time.time() - countdown_start >= COUNTDOWN_SECONDS:
                    calibration_offset = raw_angle
                    calib_curl_left = current_curl_left
                    calib_curl_right = current_curl_right
                    calib_state = CalibState.CALIBRATED
                    speed_pid.reset()
                    print("Auto-calibrated!")

            if calib_state == CalibState.CALIBRATED:
                current_angle = raw_angle - calibration_offset
                smoothed_angle = (
                    current_angle * steering_smoothing_factor
                    + smoothed_angle * (1 - steering_smoothing_factor)
                )

                delta_left = current_curl_left - calib_curl_left
                delta_right = current_curl_right - calib_curl_right

                left_open = delta_left > PROTRUSION_THRESHOLD
                right_open = delta_right > PROTRUSION_THRESHOLD

                # steering always available when both hands visible
                raw_steer_cmd = max(-1.0, min(1.0, smoothed_angle / 90.0))
                smoothed_steering_cmd = (
                    STEER_ALPHA * raw_steer_cmd
                    + (1 - STEER_ALPHA) * smoothed_steering_cmd
                )
                steer_cmd = smoothed_steering_cmd

                current_speed = get_vehicle_speed()

                # SWAPPED BEHAVIOR:
                # left hand open = hand brake
                if left_open:
                    target_drive = -1.0
                    speed_pid.reset()

                # right hand open = hold target speed with PID
                elif right_open:
                    pid_out = speed_pid.compute(current_speed, TARGET_SPEED_MPS, dt)
                    target_drive = max(-1.0, min(1.0, pid_out))

                # neither hand open = no longitudinal input
                else:
                    target_drive = 0.0
                    speed_pid.reset()

                smoothed_drive_cmd = (
                    DRIVE_ALPHA * target_drive
                    + (1 - DRIVE_ALPHA) * smoothed_drive_cmd
                )
                drive_cmd = smoothed_drive_cmd

                if now - last_print_time > PRINT_INTERVAL:
                    last_print_time = now
                    print(
                        f"SteerCmd: {steer_cmd:+.2f} | "
                        f"DriveCmd: {drive_cmd:+.2f} | "
                        f"LeftOpen: {int(left_open)} | RightOpen: {int(right_open)} | "
                        f"Speed: {current_speed:.2f} m/s | "
                        f"ΔL: {delta_left:.3f} ΔR: {delta_right:.3f}"
                    )

    # -----------------------------------------------------
    # Grace-period / reset logic
    # -----------------------------------------------------
    hands_good = (
        detection_result
        and detection_result.hand_landmarks
        and len(detection_result.hand_landmarks) == 2
        and left_hand_landmarks is not None
        and right_hand_landmarks is not None
    )

    if not hands_good:
        # no hands visible => no controller inputs
        steer_cmd = 0.0
        drive_cmd = 0.0
        speed_pid.reset()

        if hands_lost_time is None:
            hands_lost_time = time.time()

        if time.time() - hands_lost_time > LOST_THRESHOLD_SECONDS:
            if calib_state != CalibState.WAITING:
                calib_state = CalibState.WAITING
                print("Hands lost for 5 seconds — resetting to WAITING state.")
                client.reset()

    # -----------------------------------------------------
    # Manual calibration / exit keys
    # -----------------------------------------------------
    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):
        if detection_result and detection_result.hand_landmarks and len(detection_result.hand_landmarks) == 2:
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
                speed_pid.reset()
                print("Manual calibration complete!")

    if key == 27:  # ESC
        break

    # -----------------------------------------------------
    # Send control
    # -----------------------------------------------------
    send_to_vehicle(steer_cmd, drive_cmd)

    image = cv2.flip(image, 1)

    center_y = height // 2
    cv2.line(image, (0, center_y), (width, center_y), (0, 0, 255), 1)

    draw_calibration_overlay(image, calib_state, countdown_start, COUNTDOWN_SECONDS)

    current_speed = get_vehicle_speed()
    cv2.putText(image, f"Speed: {current_speed:.2f} m/s", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, f"Left open -> hold {TARGET_SPEED_MPS:.1f} m/s", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(image, "Right open -> hand brake", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(image, "No hands -> zero input", (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(image, f"SteeringCmd: {steer_cmd:+.2f}", (width - 260, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(image, f"DriveCmd: {drive_cmd:+.2f}", (width - 260, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)
    cv2.putText(image, f"LeftOpen: {int(left_open)}  RightOpen: {int(right_open)}", (width - 260, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

    cv2.imshow("MediaPipe Tasks Hands", image)

# =========================================================
# 8. CLEANUP
# =========================================================
send_to_vehicle(0.0, -1.0)
client.enableApiControl(False)

detector.close()
cap.release()
cv2.destroyAllWindows()