#!/usr/bin/env python3
"""
Game: Hug & Push Toss (Exercise 3)
Description: YOLO ball drop detection + TTS-guided hug/push exercise sets.
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
    """Dedicated TTS thread — non-blocking for vision loop."""
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

def yolo_loop(args, config):
    """Vision + exercise logic: YOLO ball drop detection."""
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

    # Open camera or video file
    cap = open_camera(args.camera)
    if not cap.isOpened():
        log_event("ERROR", "Cannot open video source", {"source": args.camera})
        sys.exit(1)

    # Config parameters
    drop_line_offset = config.get("drop_line_offset", 200)  # pixels from bottom
    target_label = config.get("target_label", "ball")
    tracker = config.get("tracker", "bytetrack.yaml")
    device = config.get("device", "cpu")
    imgsz = config.get("inference_size", 640)

    # Exercise parameters
    sets = config.get("sets", 2)
    reps = config.get("reps", 5)
    hug_time = config.get("hug_time_seconds", 5)
    push_time = config.get("push_time_seconds", 5)
    rest_time = config.get("rest_time_seconds", 45)

    # State
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    drop_line_y = frame_h - drop_line_offset
    exercise_done = False
    data_buffer = []
    frame_counter = 0

    log_event("READY", "Hug & Push Toss active", {
        "sets": sets,
        "reps": reps,
        "hug_time": hug_time,
        "push_time": push_time,
        "rest_time": rest_time,
        "drop_line": drop_line_y
    })

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                log_event("WARN", "No frame detected, ending loop")
                stop_event.set()
                break

            frame_counter += 1
            results = model.track(
                frame,
                persist=True,
                tracker=tracker,
                verbose=False,
                device=device,
                imgsz=imgsz
            )

            ball_dropped = False

            for r in results:
                if r.boxes is None:
                    continue
                boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                clss = r.boxes.cls.cpu().numpy().astype(int)

                for (x1, y1, x2, y2), cls in zip(boxes, clss):
                    label = model.names[cls] if cls < len(model.names) else "unknown"
                    if label == target_label:
                        cy = (y1 + y2) // 2
                        if cy > drop_line_y:
                            ball_dropped = True
                            log_event("DROP", "Ball dropped below line", {"frame": frame_counter})
                            break
                if ball_dropped:
                    break

            # Draw drop line
            cv2.line(frame, (0, drop_line_y), (frame.shape[1], drop_line_y), (0, 0, 255), 2)

            data_buffer.append({
                "timestamp": datetime.now().isoformat(),
                "frame": frame_counter,
                "ball_dropped": ball_dropped,
                "drop_line_y": drop_line_y
            })

            if not args.headless:
                cv2.imshow(args.game_id, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    log_event("STOP", "User pressed 'q'")
                    stop_event.set()
                    break
            else:
                time.sleep(0.01)

    except Exception as e:
        log_event("ERROR", f"YOLO loop exception: {str(e)}")
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tts_queue.put("__STOP__")
        log_event("CLEANUP", "Camera released", {"frames_processed": frame_counter})
        save_data(data_buffer, args.output, args.game_id, "hugpush")

    log_event("COMPLETED", "Hug & Push Toss ended")


def exercise_loop(args, config):
    """TTS-guided exercise sets: hug, push, rest."""
    sets = config.get("sets", 2)
    reps = config.get("reps", 5)
    hug_time = config.get("hug_time_seconds", 5)
    push_time = config.get("push_time_seconds", 5)
    rest_time = config.get("rest_time_seconds", 45)

    set_data = []  # Collect per-set metrics for CSV

    for s in range(1, sets + 1):
        if stop_event.is_set():
            break

        log_event("SET_START", f"Starting set {s}")
        tts_queue.put(f"Starting set {s}")

        set_start = datetime.now()

        for r in range(1, reps + 1):
            if stop_event.is_set():
                break

            log_event("REP", f"Set {s} - Rep {r}: Hug")
            tts_queue.put(f"Hug the ball for {hug_time} seconds")
            time.sleep(hug_time)

            if stop_event.is_set():
                break

            log_event("REP", f"Set {s} - Rep {r}: Push")
            tts_queue.put(f"Push the ball for {push_time} seconds")
            time.sleep(push_time)

        set_end = datetime.now()
        duration = (set_end - set_start).total_seconds()

        set_data.append({
            "timestamp": datetime.now().isoformat(),
            "set_number": s,
            "reps_completed": reps if not stop_event.is_set() else r,
            "start_time": set_start.isoformat(),
            "end_time": set_end.isoformat(),
            "duration_seconds": duration,
            "stopped_early": stop_event.is_set()
        })

        if s < sets and not stop_event.is_set():
            log_event("REST", f"Resting {rest_time} seconds")
            tts_queue.put(f"Take a {rest_time} seconds break")
            for _ in range(rest_time):
                if stop_event.is_set():
                    break
                time.sleep(1)

    if not stop_event.is_set():
        tts_queue.put("Exercise completed successfully! Great job.")
        time.sleep(3)  # Let TTS finish before shutting down
        stop_event.set()  # End the vision loop

    # Append set data to global buffer for saving
    return set_data


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
        # Treat as video file path
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
        description="Hug & Push Toss with YOLO Ball Drop Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python games/exe3_hugpush.py --game-id exe3 --camera 0
  python games/exe3_hugpush.py --game-id exe3 --camera "/path/to/video.mp4" --config configs/exe3_hugpush.json --headless
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

    # Start exercise logic thread
    exercise_data = []
    def exercise_wrapper():
        nonlocal exercise_data
        exercise_data = exercise_loop(args, config)

    ex_thread = threading.Thread(target=exercise_wrapper)
    ex_thread.start()

    # Run vision loop in main thread
    yolo_loop(args, config)

    # Wait for exercise thread to finish
    ex_thread.join(timeout=5.0)

    # Save exercise set metrics
    if exercise_data:
        save_data(exercise_data, args.output, args.game_id, "hugpush_sets")

    # Wait for TTS to finish
    tts_thread.join(timeout=10.0)

    log_event("COMPLETED", "Program stopped gracefully")


if __name__ == "__main__":
    main()