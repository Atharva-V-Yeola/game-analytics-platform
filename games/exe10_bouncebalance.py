#!/usr/bin/env python3
"""
Game: Bounce and Balance (Exercise 10)
Description: MediaPipe pose + YOLO ball tracking to count bounces while balancing on one leg.
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
    """Dedicated TTS thread — non-blocking."""
    if not TTS_AVAILABLE:
        log_event("INFO", "TTS not available")
        return

    engine = init_tts(config.get("tts_driver"))
    if engine is None:
        log_event("WARN", "TTS engine unavailable")
        return

    rate = config.get("tts_rate", 160)
    volume = config.get("tts_volume", 1.0)
    engine.setProperty('rate', rate)
    engine.setProperty('volume', volume)

    while not stop_event.is_set():
        try:
            msg = tts_queue.get(timeout=0.5)
            if msg == "__STOP__":
                break
            engine.say(msg)
            engine.runAndWait()
        except queue.Empty:
            continue
        except Exception as e:
            log_event("ERROR", f"TTS error: {e}")

    log_event("CLEANUP", "TTS thread ended")


# =============================================================================
# BALL VALIDATION
# =============================================================================

def is_valid_ball(box, frame_h, frame_w, config):
    """Check if detected object is likely a ball"""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    area = width * height
    cx, cy = (x1 + x2)//2, (y1 + y2)//2

    min_area_ratio = config.get("ball_min_area_ratio", 0.001)
    max_area_ratio = config.get("ball_max_area_ratio", 0.15)

    frame_area = frame_w * frame_h
    min_ball_area = frame_area * min_area_ratio
    max_ball_area = frame_area * max_area_ratio
    
    if area < min_ball_area or area > max_ball_area:
        return False, "size"

    if cx < frame_w * 0.1 or cx > frame_w * 0.9:
        return False, "edge"

    aspect_ratio = max(width, height) / max(min(width, height), 1)
    if aspect_ratio > 2.0:
        return False, "shape"

    return True, "valid"


# =============================================================================
# SETUP GUIDE
# =============================================================================

def draw_setup_guide(frame, bounce_line_ratio):
    """Draw camera setup instructions"""
    h, w = frame.shape[:2]

    cv2.putText(frame, "BOUNCE & BALANCE SETUP", (30, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(frame, "1. Stand 6-8 feet from camera", (30, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "2. Full body should be visible", (30, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "3. Ground/floor should be visible", (30, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "4. Have your ball ready", (30, 170), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    bounce_line_y = int(h * bounce_line_ratio)
    cv2.line(frame, (0, bounce_line_y), (w, bounce_line_y), (0, 255, 255), 3)
    cv2.putText(frame, "BOUNCE DETECTION LINE", (w-300, bounce_line_y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, "Press 's' when setup is ready", (30, h-30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


# =============================================================================
# MAIN GAME LOGIC
# =============================================================================

def vision_loop(args, config):
    if not ULTRALYTICS_AVAILABLE:
        log_event("ERROR", "ultralytics not installed")
        return
    if not MEDIAPIPE_AVAILABLE:
        log_event("ERROR", "mediapipe not installed")
        return

    model_path = resolve_path(config.get("model_path", args.model_path))
    if not model_path or not os.path.exists(model_path):
        log_event("ERROR", "Model weights not found", {"path": model_path})
        sys.exit(1)

    log_event("INIT", "Loading YOLO model", {"path": model_path})
    model = YOLO(model_path)

    cap = open_camera(args.camera)
    if not cap.isOpened():
        log_event("ERROR", "Cannot open video source", {"source": args.camera})
        sys.exit(1)

    # Configs
    ball_label = config.get("ball_label", "ball")
    sets_per_leg = config.get("sets_per_leg", 3)
    hold_time = config.get("hold_time_seconds", 10)
    rest_time = config.get("rest_time_seconds", 30)
    min_bounces = config.get("min_bounces", 3)
    balance_threshold = config.get("balance_threshold", 0.05)
    frame_skip = config.get("frame_skip", 2)
    ball_conf_threshold = config.get("ball_conf_threshold", 0.70)
    max_ball_lost_frames = config.get("max_ball_lost_frames", 3)
    bounce_line_ratio = config.get("bounce_line_ratio", 0.85)

    # State
    sets_done = 0
    hold_start = None
    rest_start = None
    in_rest = False
    ball_center = None
    last_ball_center = None
    ball_lost_frames = 0
    break_triggered = False
    frame_counter = 0

    bounce_count = 0
    ball_above_line = True
    last_ball_y = None

    prev_time = time.time()
    fps = 0
    data_buffer = []

    setup_mode = not args.headless
    setup_complete = False

    log_event("READY", "Bounce and Balance active", {
        "sets_per_leg": sets_per_leg,
        "hold_time": hold_time,
        "min_bounces": min_bounces,
        "bounce_line_ratio": bounce_line_ratio
    })

    if setup_mode:
        tts_queue.put("Bounce and Balance exercise. Let's set up the camera first.")

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    try:
        with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    log_event("WARN", "No frame detected, ending loop")
                    break

                frame_counter += 1
                frame_height, frame_width = frame.shape[:2]
                bounce_line_y = int(frame_height * bounce_line_ratio)

                # Setup Mode
                if setup_mode and not setup_complete:
                    draw_setup_guide(frame, bounce_line_ratio)
                    cv2.imshow(args.game_id, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s'):
                        setup_mode = False
                        setup_complete = True
                        tts_queue.put("Setup complete. Balance on one leg and bounce the ball.")
                        time.sleep(2)
                    elif key == ord('q'):
                        stop_event.set()
                        break
                    continue

                if frame_counter % 10 == 0:
                    now = time.time()
                    fps = 10 / (now - prev_time)
                    prev_time = now

                # YOLO Ball Detection
                if frame_counter % frame_skip == 0:
                    try:
                        results = model.track(frame, persist=True, verbose=False, conf=ball_conf_threshold)
                    except Exception as e:
                        log_event("ERROR", f"YOLO inference error: {e}")
                        break

                    current_ball_center = None
                    best_conf = 0

                    for r in results:
                        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                        clss = r.boxes.cls.cpu().numpy().astype(int)
                        confidences = r.boxes.conf.cpu().numpy()

                        for box, cls, conf in zip(boxes, clss, confidences):
                            label = model.names[int(cls)] if int(cls) < len(model.names) else "unknown"
                            if label == ball_label and conf > ball_conf_threshold:
                                is_valid, reason = is_valid_ball(box, frame_height, frame_width, config)
                                if is_valid:
                                    x1, y1, x2, y2 = box
                                    cx, cy = (x1 + x2)//2, (y1 + y2)//2
                                    if conf > best_conf:
                                        best_conf = conf
                                        current_ball_center = (cx, cy)

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

                # Bounce Counting
                if ball_center and not in_rest and hold_start is not None:
                    cx, cy = ball_center
                    if last_ball_y is not None:
                        if last_ball_y < bounce_line_y and cy >= bounce_line_y:
                            if ball_above_line:
                                bounce_count += 1
                                ball_above_line = False
                                tts_queue.put(f"Bounce {bounce_count}")
                                log_event("BOUNCE", "Ball bounced", {"count": bounce_count})
                        elif cy < bounce_line_y:
                            ball_above_line = True
                    last_ball_y = cy

                # MediaPipe Pose Detection
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_pose = pose.process(rgb)
                on_one_leg = False

                if results_pose.pose_landmarks:
                    if not args.headless:
                        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    landmarks = results_pose.pose_landmarks.landmark
                    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
                    diff = abs(left_ankle.y - right_ankle.y)
                    on_one_leg = diff > balance_threshold

                    if not in_rest and not break_triggered:
                        if on_one_leg and ball_center is not None:
                            if hold_start is None:
                                hold_start = time.time()
                                bounce_count = 0
                                ball_above_line = True
                                last_ball_y = None
                                tts_queue.put(f"Balance on one leg and bounce the ball. {hold_time} seconds.")

                            elapsed = time.time() - hold_start
                            remaining = max(0, int(hold_time - elapsed))

                            if not args.headless:
                                cv2.putText(frame, f"Time: {remaining}s | Bounces: {bounce_count}/{min_bounces}", 
                                          (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                cv2.putText(frame, "OK: One leg | OK: Ball detected", (30, 90), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                            if elapsed >= hold_time:
                                if bounce_count >= min_bounces:
                                    sets_done += 1
                                    tts_queue.put(f"Set {sets_done} complete! {bounce_count} bounces. Great work.")
                                    log_event("SET_COMPLETE", "Set finished successfully", {"bounces": bounce_count})
                                    hold_start = None
                                    in_rest = True
                                    rest_start = time.time()

                                    if sets_done >= sets_per_leg:
                                        tts_queue.put("All sets complete! Outstanding balance and control.")
                                        stop_event.set()
                                        break
                                else:
                                    tts_queue.put(f"Insufficient bounces. Only {bounce_count} out of {min_bounces}. Try again.")
                                    log_event("SET_FAILED", "Insufficient bounces", {"bounces": bounce_count})
                                    hold_start = None
                                    break_triggered = True
                                    in_rest = True
                                    rest_start = time.time()
                        else:
                            hold_start = None
                            bounce_count = 0
                            last_ball_y = None
                            if not args.headless:
                                if not on_one_leg:
                                    cv2.putText(frame, "Stand on ONE leg", (30, 60), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                if ball_center is None:
                                    cv2.putText(frame, "Keep ball visible", (30, 90), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    elif in_rest:
                        elapsed_rest = time.time() - rest_start
                        if not args.headless:
                            cv2.putText(frame, f"Resting... {max(0, int(rest_time - elapsed_rest))}s", 
                                      (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                        if elapsed_rest >= rest_time:
                            in_rest = False
                            tts_queue.put("Rest complete. Start next set.")
                            break_triggered = False
                            bounce_count = 0
                else:
                    if not args.headless:
                        cv2.putText(frame, "No pose detected - stand in view", (30, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Data buffer update
                data_buffer.append({
                    "timestamp": datetime.now().isoformat(),
                    "frame": frame_counter,
                    "set_number": sets_done,
                    "bounce_count": bounce_count,
                    "on_one_leg": on_one_leg,
                    "ball_visible": ball_center is not None,
                    "in_rest": in_rest,
                    "fps": fps
                })

                # Visuals
                if not args.headless:
                    cv2.line(frame, (0, bounce_line_y), (frame_width, bounce_line_y), (0, 255, 255), 3)
                    cv2.putText(frame, "BOUNCE LINE", (frame_width-150, bounce_line_y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    if ball_center:
                        cx, cy = ball_center
                        if cy >= bounce_line_y:
                            cv2.circle(frame, (cx, cy), 20, (0, 255, 255), -1)
                            cv2.circle(frame, (cx, cy), 23, (0, 200, 200), 3)
                        else:
                            cv2.circle(frame, (cx, cy), 12, (0, 255, 0), -1)
                            cv2.circle(frame, (cx, cy), 15, (0, 200, 0), 2)
                        
                        status = "BOUNCING" if cy >= bounce_line_y else "IN AIR"
                        color = (0, 255, 255) if cy >= bounce_line_y else (0, 255, 0)
                        cv2.putText(frame, status, (cx+25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    cv2.putText(frame, f"FPS: {int(fps)}", (frame_width - 120, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Sets: {sets_done}/{sets_per_leg}", 
                               (30, frame_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Total Bounces: {bounce_count}", 
                               (30, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                               (0, 255, 0) if bounce_count >= min_bounces else (0, 165, 255), 2)

                    window_title = args.game_id
                    if break_triggered:
                        window_title += " INSUFFICIENT BOUNCES"
                    
                    cv2.imshow(window_title, frame)

                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('q'):
                        stop_event.set()
                        break
                    elif key == ord('r'):
                        tts_queue.put("Resetting exercise.")
                        sets_done = 0
                        hold_start = None
                        rest_start = None
                        in_rest = False
                        ball_center = None
                        last_ball_center = None
                        ball_lost_frames = 0
                        break_triggered = False
                        bounce_count = 0
                        last_ball_y = None
                else:
                    time.sleep(0.01)

    except Exception as e:
        log_event("ERROR", f"Vision loop exception: {str(e)}")
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tts_queue.put("__STOP__")
        log_event("CLEANUP", "Camera released", {"final_sets": sets_done})
        save_data(data_buffer, args.output, args.game_id, "bouncebalance")

    log_event("COMPLETED", "Bounce & Balance ended", {"total_sets": sets_done})


# =============================================================================
# UTILITIES
# =============================================================================

def open_camera(camera_id):
    if camera_id.isdigit():
        idx = int(camera_id)
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened() and platform.system() == "Linux":
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(camera_id)
    return cap


def save_data(data_buffer, output_dir, game_id, suffix):
    if not data_buffer:
        return
    output_path = resolve_path(output_dir)
    os.makedirs(output_path, exist_ok=True)
    df = pd.DataFrame(data_buffer)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_path, f"{game_id}_{suffix}_{ts}.csv")
    df.to_csv(filepath, index=False, encoding='utf-8')
    log_event("COMPLETED", "Data saved", {"path": filepath})


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Bounce and Balance")
    parser.add_argument("--game-id", required=True, help="Unique game identifier")
    parser.add_argument("--camera", default="0", help="Camera index or video file")
    parser.add_argument("--output", default="./data", help="CSV output directory")
    parser.add_argument("--config", default=None, help="Path to JSON config")
    parser.add_argument("--model-path", default="models/all_in_one_yolov82/weights/best.pt", help="YOLO model")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")

    args = parser.parse_args()
    config = load_config(args.config)

    cv2.setNumThreads(2)
    log_event("INIT", "Game starting", {"game_id": args.game_id})

    t_tts = threading.Thread(target=tts_worker, args=(config,))
    t_tts.start()

    vision_loop(args, config)

    t_tts.join(timeout=5.0)
    log_event("COMPLETED", "Program stopped gracefully")


if __name__ == "__main__":
    main()
