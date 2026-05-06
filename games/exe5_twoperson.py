#!/usr/bin/env python3
"""
Game: Two-Person Ball Counter (Exercise 5)
Description: YOLO tracking of two persons (left/right) + ball crossing count with TTS coaching.
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

    rounds = config.get("tts_rounds", 3)
    ready_delay = config.get("tts_ready_delay", 2)
    count_delay = config.get("tts_count_delay", 1)
    break_sec = config.get("tts_break_seconds", 30)
    count_words = config.get("tts_count_words", ['One', 'Two', 'Three', 'Four', 'Five'])

    try:
        for cnt in range(rounds):
            if stop_event.is_set():
                break

            engine.say("Ready")
            engine.runAndWait()
            for _ in range(int(ready_delay * 10)):
                if stop_event.is_set():
                    break
                time.sleep(0.1)

            for word in count_words:
                if stop_event.is_set():
                    break
                engine.say(word)
                engine.runAndWait()
                for _ in range(int(count_delay * 10)):
                    if stop_event.is_set():
                        break
                    time.sleep(0.1)

            if cnt < rounds - 1 and not stop_event.is_set():
                engine.say(f"{break_sec} seconds break")
                engine.runAndWait()
                for _ in range(break_sec):
                    if stop_event.is_set():
                        break
                    time.sleep(1)

        if not stop_event.is_set():
            engine.say("Congratulations! You have successfully completed the task. Great job!")
            engine.runAndWait()

    except Exception as e:
        log_event("ERROR", f"TTS exception: {e}")
    finally:
        log_event("CLEANUP", "TTS thread ended")


# =============================================================================
# MAIN GAME LOGIC
# =============================================================================

def vision_loop(args, config):
    """YOLO tracking: two persons (left/right) + ball crossing count."""
    if not ULTRALYTICS_AVAILABLE:
        log_event("ERROR", "ultralytics not installed")
        return

    # Resolve model path
    model_path = resolve_path(config.get("model_path", args.model_path))
    if not model_path or not os.path.exists(model_path):
        log_event("ERROR", "Model weights not found", {"path": model_path})
        sys.exit(1)

    log_event("INIT", "Loading YOLO model", {"path": model_path})
    model = YOLO(model_path)

    # Open camera
    cap = open_camera(args.camera)
    if not cap.isOpened():
        log_event("ERROR", "Cannot open camera", {"camera": args.camera})
        sys.exit(1)

    # Config parameters
    line_position_ratio = config.get("line_position_ratio", 0.5)
    person_label = config.get("person_label", "person")
    ball_label = config.get("ball_label", "ball")
    tracker = config.get("tracker", "bytetrack.yaml")
    device = config.get("device", "cpu")
    imgsz = config.get("inference_size", 640)

    # State
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    line_x = int(frame_width * line_position_ratio)
    seen = set()
    count = 0
    data_buffer = []
    frame_counter = 0

    log_event("READY", "Two-Person Ball Counter active", {
        "line_x": line_x,
        "tracker": tracker,
        "device": device
    })

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                log_event("ERROR", "Frame capture failed")
                break

            frame_counter += 1
            left_person_id = None
            right_person_id = None

            # YOLO tracking
            results = model.track(
                frame,
                persist=True,
                tracker=tracker,
                verbose=False,
                device=device,
                imgsz=imgsz
            )

            for r in results:
                if r.boxes is None:
                    continue
                boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                clss = r.boxes.cls.cpu().numpy().astype(int)
                ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else []

                for (x1, y1, x2, y2), cls, oid in zip(boxes, clss, ids):
                    label = model.names[cls] if cls < len(model.names) else "unknown"
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Person detection: left vs right
                    if label == person_label:
                        if cx < line_x:
                            left_person_id = int(oid)
                            if not args.headless:
                                cv2.putText(frame, f"Left Person (ID:{oid})", (x1, y1 - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            right_person_id = int(oid)
                            if not args.headless:
                                cv2.putText(frame, f"Right Person (ID:{oid})", (x1, y1 - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Ball crossing detection
                    if label == ball_label:
                        if cx > line_x and oid not in seen:
                            count += 1
                            seen.add(oid)
                            log_event("COUNT", "Ball crossed line", {
                                "count": count,
                                "ball_id": int(oid),
                                "frame": frame_counter
                            })

                    # Draw bounding boxes
                    if not args.headless:
                        color = (0, 255, 255) if label == ball_label else (255, 255, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw line & overlay
            if not args.headless:
                cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 0, 0), 3)
                cv2.putText(frame, f"Count: {count}", (30, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.putText(frame, f"Left Person ID: {left_person_id if left_person_id else '-'}",
                           (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Right Person ID: {right_person_id if right_person_id else '-'}",
                           (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imshow(args.game_id, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    log_event("STOP", "User pressed 'q'")
                    stop_event.set()
                    break

            data_buffer.append({
                "timestamp": datetime.now().isoformat(),
                "frame": frame_counter,
                "count": count,
                "left_person_id": left_person_id,
                "right_person_id": right_person_id,
                "line_x": line_x
            })

    except Exception as e:
        log_event("ERROR", f"Vision loop exception: {str(e)}")
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()
        stop_event.set()
        log_event("CLEANUP", "Camera released", {"final_count": count})
        save_data(data_buffer, args.output, args.game_id, "twoperson")

    log_event("COMPLETED", "Two-Person Ball Counter ended", {"final_count": count})


# =============================================================================
# UTILITIES
# =============================================================================

def open_camera(camera_id):
    """Cross-platform camera opening."""
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
        description="Two-Person Ball Counter with YOLO Tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python games/exe5_twoperson.py --game-id exe5 --camera 0
  python games/exe5_twoperson.py --game-id exe5 --camera 0 --config configs/exe5_twoperson.json --headless
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

    # Launch threads
    t1 = threading.Thread(target=vision_loop, args=(args, config))
    t2 = threading.Thread(target=tts_worker, args=(config,))

    t1.start()
    t2.start()

    # Block until vision loop ends
    while t1.is_alive():
        t1.join(timeout=0.5)
        if stop_event.is_set() and t2.is_alive():
            t2.join(timeout=1.0)

    if t2.is_alive():
        t2.join(timeout=2.0)

    log_event("COMPLETED", "Program stopped gracefully")


if __name__ == "__main__":
    main()