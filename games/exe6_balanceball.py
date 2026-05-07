#!/usr/bin/env python3
"""
Game: Balance Statue with Ball (Exercise 6)
Description: MediaPipe pose + YOLO ball detection with safety zone validation and setup mode.
"""

import argparse
import cv2
import json
import numpy as np
import os
import pandas as pd
import platform
import queue
import signal
import sys
import threading
import time
from datetime import datetime

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


# =============================================================================
# GLOBALS & SIGNAL HANDLING
# =============================================================================

stop_event = threading.Event()
tts_queue = queue.Queue()

def signal_handler(signum, frame):
    stop_event.set()
    log_event("SHUTDOWN", "Signal received, stopping gracefully...")

signal.signal(signal.SIGINT, signal_handler)
if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, signal_handler)


# =============================================================================
# LOGGING
# =============================================================================

def log_event(event_type, message, data=None):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event_type,
        "message": message,
        "data": data or {}
    }
    print(json.dumps(entry), flush=True)


# =============================================================================
# PATH RESOLUTION
# =============================================================================

def get_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)

def resolve_path(path_str):
    if not path_str:
        return None
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(get_project_root(), path_str)


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(config_path):
    if config_path and os.path.exists(resolve_path(config_path)):
        with open(resolve_path(config_path), 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


# =============================================================================
# CROSS-PLATFORM TTS
# =============================================================================

def init_tts(driver=None):
    if not TTS_AVAILABLE:
        return None
    try:
        if driver:
            return pyttsx3.init(driver)
        return pyttsx3.init()
    except Exception as e:
        log_event("WARN", f"TTS init failed: {e}")
        try:
            plat = platform.system()
            if plat == 'Windows':
                return pyttsx3.init('sapi5')
            elif plat == 'Darwin':
                return pyttsx3.init('nsss')
            else:
                return pyttsx3.init('espeak')
        except Exception as e2:
            log_event("ERROR", f"TTS init failed completely: {e2}")
            return None


def tts_worker(config):
    """Dedicated TTS thread — non-blocking. COM-safe for Windows."""
    if not TTS_AVAILABLE:
        log_event("INFO", "TTS not available")
        return

    # Windows COM requires explicit per-thread initialization
    if platform.system() == "Windows":
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass

    tts_driver = config.get("tts_driver")
    rate = config.get("tts_rate", 160)
    volume = config.get("tts_volume", 1.0)

    while not stop_event.is_set():
        try:
            msg = tts_queue.get(timeout=0.5)
            if msg == "__STOP__":
                break
            # Re-init engine per utterance to prevent COM/SAPI5 deadlock on Windows
            engine = init_tts(tts_driver)
            if engine:
                engine.setProperty('rate', rate)
                engine.setProperty('volume', volume)
                engine.say(msg)
                engine.runAndWait()
                engine.stop()
                del engine
        except queue.Empty:
            continue
        except Exception as e:
            log_event("ERROR", f"TTS error: {e}")

    # Windows COM cleanup
    if platform.system() == "Windows":
        try:
            import pythoncom
            pythoncom.CoUninitialize()
        except ImportError:
            pass

    log_event("CLEANUP", "TTS thread ended")


# =============================================================================
# BALL VALIDATION (Extracted from original is_valid_ball)
# =============================================================================

def validate_ball(box, frame_h, frame_w, config):
    """
    Check if detected object is likely a valid ball.
    Returns: (is_valid: bool, reason: str)
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    area = width * height
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    min_area_ratio = config.get("ball_min_area_ratio", 0.002)
    max_area_ratio = config.get("ball_max_area_ratio", 0.08)
    min_y_ratio = config.get("ball_min_y_ratio", 0.25)
    max_y_ratio = config.get("ball_max_y_ratio", 0.75)
    max_aspect_ratio = config.get("ball_max_aspect_ratio", 1.8)

    frame_area = frame_w * frame_h
    min_ball_area = frame_area * min_area_ratio
    max_ball_area = frame_area * max_area_ratio

    # Size filtering
    if area < min_ball_area or area > max_ball_area:
        return False, "size"

    # Position filtering (avoid face/sock detection)
    if cy < frame_h * min_y_ratio:
        return False, "too_high"
    if cy > frame_h * max_y_ratio:
        return False, "too_low"

    # Shape filtering (should be roughly circular)
    aspect_ratio = max(width, height) / max(min(width, height), 1)
    if aspect_ratio > max_aspect_ratio:
        return False, "shape"

    return True, "valid"


# =============================================================================
# MAIN GAME LOGIC
# =============================================================================

def vision_loop(args, config):
    """Combined YOLO ball + MediaPipe pose with safety zone validation."""
    if not ULTRALYTICS_AVAILABLE:
        log_event("ERROR", "ultralytics not installed")
        return
    if not MEDIAPIPE_AVAILABLE:
        log_event("ERROR", "mediapipe not installed")
        return

    # Resolve model path
    model_path = resolve_path(config.get("model_path", args.model_path))
    if not model_path or not os.path.exists(model_path):
        log_event("ERROR", "Model weights not found", {"path": model_path})
        sys.exit(1)

    log_event("INIT", "Loading YOLO model", {"path": model_path})
    model = YOLO(model_path)

    # Open camera or video file
    cap = open_camera(args.camera)
    if not cap.isOpened():
        log_event("ERROR", "Cannot open video source", {"source": args.camera})
        sys.exit(1)

    # Config parameters
    ball_label = config.get("ball_label", "ball")
    ball_conf_threshold = config.get("ball_conf_threshold", 0.75)
    frame_skip = config.get("frame_skip", 3)
    max_ball_lost_frames = config.get("max_ball_lost_frames", 5)

    sets_per_leg = config.get("sets_per_leg", 3)
    hold_time = config.get("hold_time_seconds", 20)
    rest_time = config.get("rest_time_seconds", 30)
    balance_threshold = config.get("balance_threshold", 0.05)

    detection_conf = config.get("pose_detection_confidence", 0.6)
    tracking_conf = config.get("pose_tracking_confidence", 0.6)

    break_line_ratio = config.get("break_line_ratio", 0.7)
    safe_zone_left_ratio = config.get("safe_zone_left_ratio", 0.2)
    safe_zone_right_ratio = config.get("safe_zone_right_ratio", 0.8)
    safe_zone_top_ratio = config.get("safe_zone_top_ratio", 0.25)
    safe_zone_bottom_ratio = config.get("safe_zone_bottom_ratio", 0.75)

    # State
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    break_line_y = int(frame_h * break_line_ratio)

    sets_done = 0
    hold_start = None
    rest_start = None
    in_rest = False
    break_triggered = False

    ball_center = None
    last_ball_center = None
    ball_lost_frames = 0

    data_buffer = []
    frame_counter = 0
    prev_time = time.time()
    fps = 0

    # Setup mode
    setup_mode = not args.headless  # Auto-skip setup in headless mode
    setup_complete = False
    setup_start_time = time.time()

    log_event("READY", "Balance with Ball active", {
        "sets_per_leg": sets_per_leg,
        "hold_time": hold_time,
        "rest_time": rest_time,
        "frame_skip": frame_skip,
        "setup_mode": setup_mode
    })

    # MediaPipe pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    try:
        with mp_pose.Pose(min_detection_confidence=detection_conf,
                         min_tracking_confidence=tracking_conf) as pose:

            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    log_event("WARN", "No frame detected, ending loop")
                    break

                frame_counter += 1

                # --------------------------
                # SETUP MODE
                # --------------------------
                if setup_mode and not setup_complete:
                    # Auto-start after 5 seconds (for backend-launched processes)
                    if time.time() - setup_start_time >= 5:
                        setup_mode = False
                        setup_complete = True
                        tts_queue.put("Auto-starting. Get ready to balance on one leg.")
                        time.sleep(2)
                        continue

                    draw_setup_guide(frame, frame_h, frame_w, break_line_y)
                    cv2.imshow(args.game_id, frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s'):
                        setup_mode = False
                        setup_complete = True
                        tts_queue.put("Setup complete. Get ready to balance on one leg.")
                        time.sleep(2)
                    elif key == ord('q'):
                        stop_event.set()
                        break
                    continue

                # --------------------------
                # FPS CALCULATION
                # --------------------------
                if frame_counter % 10 == 0:
                    now = time.time()
                    fps = 10 / (now - prev_time)
                    prev_time = now

                # --------------------------
                # YOLO BALL DETECTION
                # --------------------------
                if frame_counter % frame_skip == 0:
                    try:
                        results_yolo = model.track(
                            frame,
                            persist=True,
                            verbose=False,
                            conf=ball_conf_threshold
                        )
                    except Exception as e:
                        log_event("ERROR", f"YOLO inference error: {e}")
                        stop_event.set()
                        break

                    current_ball_center = None
                    best_conf = 0

                    for r in results_yolo:
                        if r.boxes is None:
                            continue
                        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                        clss = r.boxes.cls.cpu().numpy().astype(int)
                        confidences = r.boxes.conf.cpu().numpy()

                        for box, cls, conf in zip(boxes, clss, confidences):
                            label = model.names[int(cls)] if int(cls) < len(model.names) else "unknown"
                            if label == ball_label and conf > ball_conf_threshold:
                                is_valid, reason = validate_ball(box, frame_h, frame_w, config)
                                if is_valid:
                                    x1, y1, x2, y2 = box
                                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                                    if conf > best_conf:
                                        best_conf = conf
                                        current_ball_center = (cx, cy)

                    # Update ball tracking state
                    if current_ball_center is not None:
                        ball_center = current_ball_center
                        last_ball_center = current_ball_center
                        ball_lost_frames = 0
                    else:
                        ball_lost_frames += 1
                        if ball_lost_frames <= max_ball_lost_frames and last_ball_center is not None:
                            ball_center = last_ball_center
                        else:
                            ball_center = None

                # --------------------------
                # BALL BREAK RULES
                # --------------------------
                if ball_center and not in_rest:
                    cx, cy = ball_center
                    safe_left = int(frame_w * safe_zone_left_ratio)
                    safe_right = int(frame_w * safe_zone_right_ratio)

                    # Break: Ball drops below line
                    if cy >= break_line_y and not break_triggered:
                        tts_queue.put("Ball dropped below safety line! Stopping this set.")
                        break_triggered = True
                        ball_center = None

                    # Break: Ball outside safe zone horizontally
                    elif (cx < safe_left or cx > safe_right) and not break_triggered:
                        tts_queue.put("Ball moved outside safe area. Stopping this set.")
                        break_triggered = True
                        ball_center = None

                # --------------------------
                # MEDIAPIPE POSE DETECTION
                # --------------------------
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_pose = pose.process(rgb)

                on_one_leg = False

                if results_pose.pose_landmarks:
                    if not args.headless:
                        mp_drawing.draw_landmarks(
                            frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS
                        )

                    landmarks = results_pose.pose_landmarks.landmark
                    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
                    diff = abs(left_ankle.y - right_ankle.y)
                    on_one_leg = diff > balance_threshold

                    # --------------------------
                    # MAIN LOGIC
                    # --------------------------
                    if not in_rest and not break_triggered:
                        if on_one_leg and ball_center is not None:
                            if hold_start is None:
                                hold_start = time.time()
                                tts_queue.put("Hold steady for twenty seconds. Keep the ball in the safe zone.")

                            elapsed = time.time() - hold_start
                            remaining = max(0, int(hold_time - elapsed))

                            if not args.headless:
                                cv2.putText(frame, f"Holding... {remaining}s", (30, 60),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                cv2.putText(frame, "Ball detected", (30, 90),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                            if elapsed >= hold_time:
                                sets_done += 1
                                tts_queue.put(f"Set {sets_done} complete! Excellent balance.")
                                hold_start = None
                                in_rest = True
                                rest_start = time.time()

                                data_buffer.append({
                                    "timestamp": datetime.now().isoformat(),
                                    "event": "set_complete",
                                    "set_number": sets_done,
                                    "hold_time": hold_time
                                })

                                if sets_done >= sets_per_leg:
                                    tts_queue.put("All sets complete! Outstanding work.")
                                    stop_event.set()
                                    break
                        else:
                            hold_start = None
                            if not args.headless:
                                if not on_one_leg:
                                    cv2.putText(frame, "Stand on ONE leg", (30, 60),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                if ball_center is None:
                                    cv2.putText(frame, "Keep ball visible", (30, 90),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    elif in_rest:
                        elapsed_rest = time.time() - rest_start
                        remaining_rest = max(0, int(rest_time - elapsed_rest))

                        if not args.headless:
                            cv2.putText(frame, f"Resting... {remaining_rest}s", (30, 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                        if elapsed_rest >= rest_time:
                            in_rest = False
                            tts_queue.put("Rest complete. Start next set.")
                            break_triggered = False
                            hold_start = None

                else:
                    if not args.headless:
                        cv2.putText(frame, "No pose detected - stand in view", (30, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # --------------------------
                # VISUALS & FEEDBACK
                # --------------------------
                if not args.headless:
                    # Safety lines
                    cv2.line(frame, (0, break_line_y), (frame_w, break_line_y), (0, 0, 255), 2)
                    cv2.putText(frame, "BALL DROP LINE", (frame_w - 180, break_line_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

                    # Safe zone
                    safe_top = int(frame_h * safe_zone_top_ratio)
                    safe_bottom = int(frame_h * safe_zone_bottom_ratio)
                    safe_left = int(frame_w * safe_zone_left_ratio)
                    safe_right = int(frame_w * safe_zone_right_ratio)
                    cv2.rectangle(frame, (safe_left, safe_top), (safe_right, safe_bottom), (255, 255, 0), 2)
                    cv2.putText(frame, "SAFE ZONE", (safe_left + 10, safe_top + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

                    # Draw ball
                    if ball_center:
                        cx, cy = ball_center
                        cv2.circle(frame, (cx, cy), 12, (0, 255, 255), -1)
                        cv2.circle(frame, (cx, cy), 15, (0, 200, 200), 2)

                        status = "VALID BALL" if cy < break_line_y else "TOO LOW"
                        color = (0, 255, 0) if status == "VALID BALL" else (0, 165, 255)
                        cv2.putText(frame, status, (cx + 20, cy - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Status overlay
                    cv2.putText(frame, f"FPS: {int(fps)}", (frame_w - 120, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Sets: {sets_done}/{sets_per_leg}",
                               (30, frame_h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    ball_status = "DETECTED" if ball_center else "LOST"
                    ball_color = (0, 255, 0) if ball_center else (0, 0, 255)
                    cv2.putText(frame, f"Ball: {ball_status}",
                               (30, frame_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ball_color, 2)

                    # Window title
                    title = args.game_id
                    if break_triggered:
                        title += " - BREAK TRIGGERED"
                    cv2.imshow(title, frame)

                    # Manual controls
                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('q'):
                        stop_event.set()
                        break
                    elif key == ord('r'):
                        tts_queue.put("Resetting game.")
                        sets_done = 0
                        hold_start = None
                        rest_start = None
                        in_rest = False
                        ball_center = None
                        last_ball_center = None
                        ball_lost_frames = 0
                        break_triggered = False

                # Frame metrics
                data_buffer.append({
                    "timestamp": datetime.now().isoformat(),
                    "frame": frame_counter,
                    "sets_done": sets_done,
                    "in_rest": in_rest,
                    "break_triggered": break_triggered,
                    "ball_detected": ball_center is not None,
                    "on_one_leg": on_one_leg,
                    "fps": fps
                })

    except Exception as e:
        log_event("ERROR", f"Vision loop exception: {str(e)}")
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tts_queue.put("__STOP__")
        log_event("CLEANUP", "Camera released", {"final_sets": sets_done})
        save_data(data_buffer, args.output, args.game_id, "balanceball")

    log_event("COMPLETED", "Balance with Ball ended", {"total_sets": sets_done})


# =============================================================================
# SETUP GUIDE DRAWING
# =============================================================================

def draw_setup_guide(frame, frame_h, frame_w, break_line_y):
    """Draw camera setup instructions on frame."""
    cv2.putText(frame, "CAMERA SETUP GUIDE", (30, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(frame, "1. Stand 6-8 feet from camera", (30, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "2. Camera at eye level - no tilt down", (30, 110),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "3. Frame should show throat to knees", (30, 140),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "4. Face and feet should NOT be visible", (30, 170),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "5. Ball should be clearly visible", (30, 200),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Guide lines
    cv2.line(frame, (0, int(frame_h * 0.2)), (frame_w, int(frame_h * 0.2)), (0, 255, 255), 2)
    cv2.putText(frame, "TOP: Throat level", (frame_w - 250, int(frame_h * 0.2) - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.line(frame, (0, int(frame_h * 0.8)), (frame_w, int(frame_h * 0.8)), (0, 255, 0), 2)
    cv2.putText(frame, "BOTTOM: Just above ankles", (frame_w - 280, int(frame_h * 0.8) - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.rectangle(frame, (0, int(frame_h * 0.3)), (frame_w, int(frame_h * 0.7)), (255, 255, 0), 2)
    cv2.putText(frame, "BALL SAFE ZONE", (frame_w - 200, int(frame_h * 0.5)),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(frame, "Press 's' when setup is ready", (30, frame_h - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


# =============================================================================
# UTILITIES
# =============================================================================

def open_camera(camera_id):
    """Cross-platform camera/video opening."""
    if camera_id.isdigit():
        idx = int(camera_id)
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(idx)
        if not cap.isOpened() and platform.system() == "Linux":
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(camera_id)
    return cap


def save_data(data_buffer, output_dir, game_id, suffix):
    """Flush collected data to CSV."""
    if not data_buffer:
        log_event("COMPLETED", "No data to save")
        return

    output_path = resolve_path(output_dir)
    os.makedirs(output_path, exist_ok=True)

    df = pd.DataFrame(data_buffer)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{game_id}_{suffix}_{ts}.csv"
    filepath = os.path.join(output_path, filename)
    df.to_csv(filepath, index=False, encoding='utf-8')
    log_event("COMPLETED", "Data saved", {"path": filepath, "rows": len(df)})


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Balance Statue with Ball - YOLO + MediaPipe Pose",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python games/exe6_balanceball.py --game-id exe6 --camera 0
  python games/exe6_balanceball.py --game-id exe6 --camera "/path/to/video.mp4" --config configs/exe6_balanceball.json --headless
        """
    )
    parser.add_argument("--game-id", required=True, help="Unique game identifier")
    parser.add_argument("--camera", default="0", help="Camera index or video file path")
    parser.add_argument("--output", default="./data", help="CSV output directory")
    parser.add_argument("--config", default=None, help="Path to JSON config file")
    parser.add_argument("--model-path", default="models/all_in_one_yolov82/weights/best.pt",
                        help="Path to YOLO model weights")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI display (skips setup mode)")

    args = parser.parse_args()
    config = load_config(args.config)

    cv2.setNumThreads(2)

    log_event("INIT", "Game starting", {
        "game_id": args.game_id,
        "camera": args.camera,
        "platform": platform.system(),
        "python": sys.executable
    })

    # Start TTS thread
    tts_thread = threading.Thread(target=tts_worker, args=(config,))
    tts_thread.start()

    # Run vision loop in main thread
    vision_loop(args, config)

    # Wait for TTS to finish
    tts_thread.join(timeout=10.0)

    log_event("COMPLETED", "Program stopped gracefully")


if __name__ == "__main__":
    main()
    