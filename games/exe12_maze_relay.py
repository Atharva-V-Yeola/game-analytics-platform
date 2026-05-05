#!/usr/bin/env python3
"""
Game: Agility Maze Relay (Exercise 12)
Description: YOLO tracking for alternating zone bouncing.
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
# HELPER FUNCTIONS
# =============================================================================

def get_person_and_ball(results, model):
    person_x, person_y, person_box = None, None, None
    ball_x, ball_y, ball_box = None, None, None
    ball_visible = False

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        classes = r.boxes.cls.cpu().numpy().astype(int)

        for box, cls in zip(boxes, classes):
            name = model.names[int(cls)] if int(cls) < len(model.names) else "unknown"

            if name == "person":
                x1, y1, x2, y2 = box
                person_x = (x1 + x2) // 2
                person_y = (y1 + y2) // 2
                person_box = box
            elif name == "ball":
                x1, y1, x2, y2 = box
                ball_x = (x1 + x2) // 2
                ball_y = y2  # Bottom of ball for bounce detection
                ball_box = box
                ball_visible = True

    return person_x, person_y, person_box, ball_x, ball_y, ball_box, ball_visible


def determine_zone(x, vertical_line):
    """Determine if position is in LEFT or RIGHT zone"""
    if x is None:
        return None
    return "LEFT" if x < vertical_line else "RIGHT"


# =============================================================================
# MAIN GAME LOGIC
# =============================================================================

def vision_loop(args, config):
    if not ULTRALYTICS_AVAILABLE:
        log_event("ERROR", "ultralytics not installed")
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
    sets = config.get("sets", 3)
    bounces_per_zone = config.get("bounces_per_zone", 5)
    rest_between_sets = config.get("rest_between_sets_seconds", 90)
    yolo_conf_threshold = config.get("yolo_conf_threshold", 0.5)
    left_horizontal_ratio = config.get("left_horizontal_ratio", 0.6)
    right_horizontal_ratio = config.get("right_horizontal_ratio", 0.6)

    # State
    current_set = 1
    left_count = 0
    right_count = 0
    last_bounce_zone = None
    ball_was_above_line = True
    status_msg = "Start bouncing!"
    
    is_resting = False
    rest_start_time = 0
    
    frame_counter = 0
    data_buffer = []
    violation_pause_until = 0

    log_event("READY", "Agility Maze Relay active", {
        "sets": sets,
        "bounces_per_zone": bounces_per_zone
    })

    tts_queue.put("Agility Maze Relay. Navigate between zones while bouncing the ball. You must alternate zones!")
    time.sleep(2)
    tts_queue.put(f"Set {current_set}. Start!")

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                log_event("WARN", "No frame detected, ending loop")
                break

            frame_counter += 1
            h, w = frame.shape[:2]
            current_time = time.time()

            vertical_line = w // 2
            left_horizontal = int(h * left_horizontal_ratio)
            right_horizontal = int(h * right_horizontal_ratio)

            # State handling: REST
            if is_resting:
                remaining = int(rest_between_sets - (current_time - rest_start_time))
                if remaining > 0:
                    if remaining % 10 == 0 and remaining < rest_between_sets:
                        if getattr(vision_loop, "last_rest_announcement", None) != remaining:
                            # only log/announce occasionally
                            if remaining <= 10 or remaining % 30 == 0:
                                tts_queue.put(f"{remaining} seconds")
                            vision_loop.last_rest_announcement = remaining
                            
                    if not args.headless:
                        cv2.putText(frame, f"REST: {remaining}s", (w//2 - 150, h//2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
                        cv2.imshow(args.game_id, frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            stop_event.set()
                            break
                    else:
                        time.sleep(0.01)
                    continue
                else:
                    is_resting = False
                    tts_queue.put(f"Set {current_set}. Start!")
                    vision_loop.last_rest_announcement = None

            # Handle Violation Pause
            if violation_pause_until > current_time:
                if not args.headless:
                    cv2.putText(frame, "VIOLATION!", (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    cv2.imshow(args.game_id, frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_event.set()
                        break
                else:
                    time.sleep(0.01)
                continue
            elif violation_pause_until > 0 and violation_pause_until <= current_time:
                # Reset after violation
                violation_pause_until = 0
                left_count = 0
                right_count = 0
                last_bounce_zone = None
                status_msg = "Start bouncing!"
                tts_queue.put(f"Set {current_set} failed. Let's try again!")
                log_event("RESTART", f"Restarting set {current_set} after violation")

            # YOLO detection
            try:
                results = model(frame, conf=yolo_conf_threshold, verbose=False)
            except Exception as e:
                log_event("ERROR", f"YOLO inference error: {e}")
                break

            person_x, person_y, person_box, ball_x, ball_y, ball_box, ball_visible = get_person_and_ball(results, model)

            person_zone = determine_zone(person_x, vertical_line)
            ball_zone = determine_zone(ball_x, vertical_line)

            # Bounce logic
            if ball_visible and ball_x is not None and ball_y is not None:
                horizontal_line = left_horizontal if ball_zone == "LEFT" else (right_horizontal if ball_zone == "RIGHT" else None)

                if horizontal_line:
                    ball_below_line = ball_y > horizontal_line

                    if ball_below_line and ball_was_above_line:
                        if person_zone == ball_zone:
                            if ball_zone == last_bounce_zone:
                                status_msg = "FAIL: Consecutive bounces!"
                                log_event("VIOLATION", "Consecutive bounces in same zone", {"zone": ball_zone})
                                tts_queue.put("Violation! You must alternate between zones!")
                                violation_pause_until = current_time + 2.0
                            else:
                                if ball_zone == "LEFT":
                                    if left_count < bounces_per_zone:
                                        left_count += 1
                                        status_msg = f"LEFT bounce #{left_count} - Switch to RIGHT!"
                                        log_event("BOUNCE", "Left zone bounce", {"count": left_count})
                                else:
                                    if right_count < bounces_per_zone:
                                        right_count += 1
                                        status_msg = f"RIGHT bounce #{right_count} - Switch to LEFT!"
                                        log_event("BOUNCE", "Right zone bounce", {"count": right_count})

                                last_bounce_zone = ball_zone

                                if left_count > bounces_per_zone or right_count > bounces_per_zone:
                                    status_msg = "FAIL: Too many bounces!"
                                    log_event("VIOLATION", "Too many bounces in one zone")
                                    tts_queue.put("Violation! Too many bounces in one zone!")
                                    violation_pause_until = current_time + 2.0

                                if left_count == bounces_per_zone and right_count == bounces_per_zone:
                                    status_msg = "SET COMPLETE!"
                                    log_event("SET_COMPLETE", f"Set {current_set} complete")
                                    tts_queue.put(f"Set {current_set} complete! Excellent work!")
                                    
                                    if not args.headless:
                                        cv2.putText(frame, "SET COMPLETE!", (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                                        cv2.imshow(args.game_id, frame)
                                        cv2.waitKey(2000)
                                    
                                    current_set += 1
                                    left_count = 0
                                    right_count = 0
                                    last_bounce_zone = None
                                    
                                    if current_set <= sets:
                                        tts_queue.put(f"Rest for {rest_between_sets} seconds")
                                        is_resting = True
                                        rest_start_time = time.time()
                                    else:
                                        tts_queue.put("All sets complete! Outstanding performance!")
                                        log_event("COMPLETED", "All sets finished")
                                        stop_event.set()
                                        break

                    ball_was_above_line = not ball_below_line

            # Data buffer
            data_buffer.append({
                "timestamp": datetime.now().isoformat(),
                "frame": frame_counter,
                "set": current_set,
                "left_count": left_count,
                "right_count": right_count,
                "person_zone": person_zone,
                "ball_zone": ball_zone,
                "last_bounce_zone": last_bounce_zone
            })

            # Visuals
            if not args.headless and not is_resting and violation_pause_until == 0:
                cv2.line(frame, (vertical_line, 0), (vertical_line, h), (255, 255, 0), 3)
                cv2.line(frame, (0, left_horizontal), (vertical_line, left_horizontal), (0, 255, 0), 3)
                cv2.line(frame, (vertical_line, right_horizontal), (w, right_horizontal), (0, 0, 255), 3)
                
                if person_box is not None:
                    x1, y1, x2, y2 = person_box
                    color = (255, 0, 0) if person_zone == "LEFT" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                if ball_box is not None:
                    x1, y1, x2, y2 = ball_box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                cv2.rectangle(frame, (10, h - 220), (400, h - 10), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, h - 220), (400, h - 10), (255, 255, 255), 2)

                y = h - 190
                cv2.putText(frame, f"Set: {current_set}/{sets}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                left_color = (0, 255, 0) if left_count == bounces_per_zone else (255, 255, 255)
                cv2.putText(frame, f"LEFT: {left_count}/{bounces_per_zone}", (20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, left_color, 2)
                right_color = (0, 255, 0) if right_count == bounces_per_zone else (255, 255, 255)
                cv2.putText(frame, f"RIGHT: {right_count}/{bounces_per_zone}", (20, y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, right_color, 2)

                if last_bounce_zone:
                    zone_color = (255, 0, 0) if last_bounce_zone == "LEFT" else (0, 0, 255)
                    cv2.putText(frame, f"Last: {last_bounce_zone}", (20, y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, zone_color, 2)
                if status_msg:
                    cv2.putText(frame, status_msg, (20, y + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imshow(args.game_id, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break

    except Exception as e:
        log_event("ERROR", f"Vision loop exception: {str(e)}")
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tts_queue.put("__STOP__")
        log_event("CLEANUP", "Camera released")
        save_data(data_buffer, args.output, args.game_id, "maze_relay")

    log_event("COMPLETED", "Agility Maze Relay ended")


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
    parser = argparse.ArgumentParser(description="Agility Maze Relay")
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
