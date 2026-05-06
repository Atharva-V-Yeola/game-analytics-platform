#!/usr/bin/env python3
"""
Game: Walk & Tap Relay (Exercise 4)
Description: YOLO detection of person + ball in a region, with TTS-guided sets.
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
# MAIN GAME LOGIC
# =============================================================================

def vision_loop(args, config):
    """YOLO detection: person + ball in region."""
    if not ULTRALYTICS_AVAILABLE:
        log_event("ERROR", "ultralytics not installed")
        return

    # Resolve model path (NOTE: exe4 uses different model folder than exe1/3)
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
    region_left_ratio = config.get("region_left_ratio", 0.33)
    region_right_ratio = config.get("region_right_ratio", 0.67)
    person_label = config.get("person_label", "person")
    ball_label = config.get("ball_label", "ball")
    tracker = config.get("tracker", "bytetrack.yaml")
    device = config.get("device", "cpu")
    imgsz = config.get("inference_size", 640)
    use_tracking = config.get("use_tracking", False)  # exe4 uses detection, not tracking

    # Exercise parameters
    sets = config.get("sets", 2)
    rest_time = config.get("rest_time_seconds", 60)

    # State
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    left_x = int(frame_w * region_left_ratio)
    right_x = int(frame_w * region_right_ratio)

    data_buffer = []
    frame_counter = 0
    current_set = 0
    set_results = []

    log_event("READY", "Walk & Tap Relay active", {
        "sets": sets,
        "rest_time": rest_time,
        "region": [left_x, right_x],
        "use_tracking": use_tracking
    })

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                log_event("WARN", "No frame detected, ending loop")
                stop_event.set()
                break

            frame_counter += 1

            # YOLO inference
            try:
                if use_tracking:
                    results_infer = model.track(
                        frame,
                        persist=True,
                        tracker=tracker,
                        verbose=False,
                        device=device,
                        imgsz=imgsz
                    )
                else:
                    results_infer = model(
                        frame,
                        verbose=False,
                        device=device,
                        imgsz=imgsz
                    )
            except Exception as e:
                log_event("ERROR", f"Model inference error: {e}")
                stop_event.set()
                break

            # Detection logic
            person_detected = False
            ball_detected = False
            person_in_region = False
            ball_in_region = False

            for r in results_infer:
                if r.boxes is None:
                    continue
                boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                clss = r.boxes.cls.cpu().numpy().astype(int)

                for (x1, y1, x2, y2), cls in zip(boxes, clss):
                    label = model.names[cls] if cls < len(model.names) else "unknown"
                    cx = (x1 + x2) // 2

                    if label == person_label:
                        person_detected = True
                        if left_x < cx < right_x:
                            person_in_region = True
                            if not args.headless:
                                cv2.circle(frame, (cx, (y1+y2)//2), 8, (255, 0, 0), -1)

                    elif label == ball_label:
                        ball_detected = True
                        if left_x < cx < right_x:
                            ball_in_region = True
                            if not args.headless:
                                cv2.circle(frame, (cx, (y1+y2)//2), 5, (0, 255, 0), -1)

            both_in_region = person_detected and ball_detected and person_in_region and ball_in_region

            # Metrics
            frame_metrics = {
                "timestamp": datetime.now().isoformat(),
                "frame": frame_counter,
                "person_detected": person_detected,
                "ball_detected": ball_detected,
                "both_in_region": both_in_region,
                "left_x": left_x,
                "right_x": right_x
            }

            # Draw region
            if not args.headless:
                cv2.line(frame, (left_x, 0), (left_x, frame_h), (255, 0, 0), 2)
                cv2.line(frame, (right_x, 0), (right_x, frame_h), (255, 0, 0), 2)

                if both_in_region:
                    cv2.rectangle(frame, (left_x, 0), (right_x, frame_h), (0, 255, 255), 3)
                    cv2.putText(frame, "IN REGION - Keep going!", (50, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.putText(frame, f"Status: {'ACTIVE' if both_in_region else 'INACTIVE'}", (50, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow(args.game_id, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    log_event("STOP", "User pressed 'q'")
                    stop_event.set()
                    break

            data_buffer.append(frame_metrics)

            if frame_counter % 30 == 0:
                log_event("STATUS", f"Frame {frame_counter}, both_in_region={both_in_region}")

    except Exception as e:
        log_event("ERROR", f"Vision loop exception: {str(e)}")
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tts_queue.put("__STOP__")
        log_event("CLEANUP", "Camera released", {"frames_processed": frame_counter})
        save_data(data_buffer, args.output, args.game_id, "walktap")

    log_event("COMPLETED", "Walk & Tap Relay ended")


def exercise_coordinator(args, config):
    """TTS-guided set coordination."""
    sets = config.get("sets", 2)
    rest_time = config.get("rest_time_seconds", 60)

    for s in range(1, sets + 1):
        if stop_event.is_set():
            break

        log_event("SET_START", f"Starting set {s}")
        tts_queue.put(f"Walk & Tap Relay - Set {s} start. Walk to cone with ball, tap it, turn around and return.")

        # Wait for set duration (simplified: user performs while vision loop runs)
        # In original code, yolo_loop blocked here. Now vision loop runs independently.
        # We just signal the set number and let vision loop capture data.
        time.sleep(2)  # Brief pause for TTS to finish

        if s < sets and not stop_event.is_set():
            log_event("REST", f"Resting {rest_time} seconds")
            tts_queue.put(f"Take a {rest_time} seconds break before next set.")
            for _ in range(rest_time):
                if stop_event.is_set():
                    break
                time.sleep(1)

    if not stop_event.is_set():
        tts_queue.put("Congratulations! Walk & Tap Relay drill completed successfully.")
        time.sleep(3)
        stop_event.set()


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
        description="Walk & Tap Relay with YOLO Person+Ball Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python games/exe4_walktap.py --game-id exe4 --camera 0
  python games/exe4_walktap.py --game-id exe4 --camera "/path/to/video.mp4" --config configs/exe4_walktap.json --headless
        """
    )
    parser.add_argument("--game-id", required=True, help="Unique game identifier")
    parser.add_argument("--camera", default="0", help="Camera index or video file path")
    parser.add_argument("--output", default="./data", help="CSV output directory")
    parser.add_argument("--config", default=None, help="Path to JSON config file")
    parser.add_argument("--model-path", default="models/all_in_one_yolov82/weights/best.pt",
                        help="Path to YOLO model weights")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI display")

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

    # Start exercise coordinator thread
    coord_thread = threading.Thread(target=exercise_coordinator, args=(args, config))
    coord_thread.start()

    # Run vision loop in main thread
    vision_loop(args, config)

    # Wait for coordinator to finish
    coord_thread.join(timeout=5.0)

    # Wait for TTS to finish
    tts_thread.join(timeout=10.0)

    log_event("COMPLETED", "Program stopped gracefully")


if __name__ == "__main__":
    main()