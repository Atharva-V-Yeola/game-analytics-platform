#!/usr/bin/env python3
"""
Game: YOLO Ball Counter (Exercise 1)
Description: Tracks balls crossing a vertical line using YOLOv8 with audio coaching.
"""

import argparse
import cv2
import json
import numpy as np
import os
import pandas as pd
import platform
import signal
import sys
import threading
import time
from datetime import datetime

# Optional imports with graceful fallback
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
# CROSS-PLATFORM TTS
# =============================================================================

def init_tts(driver=None):
    """Initialize TTS engine with cross-platform fallback."""
    if not TTS_AVAILABLE:
        return None
    try:
        if driver:
            return pyttsx3.init(driver)
        return pyttsx3.init()
    except Exception as e:
        log_event("WARN", f"TTS default init failed: {e}")
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


# =============================================================================
# PATH RESOLUTION
# =============================================================================

def get_project_root():
    """
    Resolve project root directory.
    Script lives in games/ folder, so project root is one level up.
    This ensures paths work regardless of where the script is launched from.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    return project_root


def resolve_path(path_str):
    """
    Convert a path string to an absolute path.
    If relative, resolve against project root.
    """
    if not path_str:
        return None
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(get_project_root(), path_str)


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(config_path):
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


# =============================================================================
# MAIN GAME LOGIC
# =============================================================================

def yolo_loop(args, config):
    """Vision loop: YOLO detection + line crossing count."""
    if not ULTRALYTICS_AVAILABLE:
        log_event("ERROR", "ultralytics not installed")
        return

    # Resolve model path against project root
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

    # Tunable parameters
    line_x = config.get("line_x", 300)
    target_label = config.get("target_label", "ball")
    frame_skip = config.get("frame_skip", 1)
    display_w = config.get("frame_width", 640)
    display_h = config.get("frame_height", 480)
    tracker = config.get("tracker", "bytetrack.yaml")
    device = config.get("device", "cpu")
    half = config.get("half", False)
    imgsz = config.get("inference_size", 640)

    # State
    seen = set()
    count = 0
    data_buffer = []
    frame_counter = 0

    log_event("READY", "YOLO loop active", {
        "line_x": line_x,
        "target": target_label,
        "frame_skip": frame_skip,
        "device": device
    })

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                log_event("ERROR", "Frame capture failed")
                break

            frame_counter += 1

            # Resize for consistent processing & display
            display_frame = cv2.resize(frame, (display_w, display_h))

            # Scale line position if original calibration differs from display size
            scale_x = display_frame.shape[1] / frame.shape[1] if frame.shape[1] > 0 else 1.0
            scaled_line_x = int(line_x * scale_x)

            # YOLO inference (honor frame skip to reduce CPU load)
            if frame_counter % frame_skip == 0:
                results = model.track(
                    display_frame,
                    persist=True,
                    tracker=tracker,
                    verbose=False,
                    device=device,
                    half=half,
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
                        if label == target_label:
                            cx = (x1 + x2) // 2
                            if cx > scaled_line_x and oid not in seen:
                                count += 1
                                seen.add(oid)
                                log_event("COUNT", "Ball crossed line", {
                                    "count": count,
                                    "ball_id": int(oid),
                                    "frame": frame_counter
                                })
                                data_buffer.append({
                                    "timestamp": datetime.now().isoformat(),
                                    "event": "crossing",
                                    "ball_id": int(oid),
                                    "count": count,
                                    "line_x": line_x,
                                    "frame": frame_counter
                                })

            # Draw overlay
            cv2.line(display_frame, (scaled_line_x, 0), (scaled_line_x, display_frame.shape[0]), (255, 0, 0), 2)
            cv2.putText(display_frame, f"Count: {count}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

            if not args.headless:
                cv2.imshow("YOLO Ball Counter", display_frame)
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
        log_event("CLEANUP", "Camera released", {"final_count": count})
        save_data(data_buffer, args.output, args.game_id, "crossings")

    log_event("COMPLETED", "YOLO loop ended", {"final_count": count})


def tts_loop(args, config):
    """Audio coaching loop."""
    if args.headless or not TTS_AVAILABLE:
        log_event("INFO", "TTS skipped (headless or unavailable)")
        return

    engine = init_tts(config.get("tts_driver"))
    if engine is None:
        log_event("WARN", "TTS engine unavailable")
        return

    rounds = config.get("tts_rounds", 3)
    break_sec = config.get("tts_break_seconds", 30)
    count_words = config.get("tts_count_words", ['One', 'Two', 'Three', 'Four', 'Five'])

    try:
        for cnt in range(rounds):
            if stop_event.is_set():
                break

            engine.say("Ready")
            engine.runAndWait()
            time.sleep(2)

            for word in count_words:
                if stop_event.is_set():
                    break
                engine.say(word)
                engine.runAndWait()
                time.sleep(1)

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
        log_event("CLEANUP", "TTS loop ended")


# =============================================================================
# UTILITIES
# =============================================================================

def open_camera(camera_id):
    """Cross-platform camera opening with fallback."""
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
    """Flush collected events to CSV."""
    if not data_buffer:
        log_event("COMPLETED", "No crossing data to save")
        return

    # Resolve output directory against project root if relative
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
        description="YOLO Ball Counter with TTS Coaching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python games/exe1_yolo_ball.py --game-id yolo_ball --camera 0
  python games/exe1_yolo_ball.py --game-id yolo_ball --camera 0 --config configs/exe1_yolo_ball.json --headless
        """
    )
    parser.add_argument("--game-id", required=True, help="Unique game identifier")
    parser.add_argument("--camera", default="0", help="Camera index or video file path")
    parser.add_argument("--output", default="./data", help="CSV output directory")
    parser.add_argument("--config", default=None, help="Path to JSON config file")
    parser.add_argument("--model-path", default="models/all_in_one_yolov82/weights/best.pt",
                        help="Path to YOLO model weights")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI display (disables TTS and cv2.imshow)")

    args = parser.parse_args()
    config = load_config(resolve_path(args.config))

    # Limit OpenCV threading to prevent CPU starvation
    cv2.setNumThreads(2)

    log_event("INIT", "Game starting", {
        "game_id": args.game_id,
        "camera": args.camera,
        "platform": platform.system(),
        "python": sys.executable
    })

    # Launch threads
    t1 = threading.Thread(target=yolo_loop, args=(args, config))
    t2 = threading.Thread(target=tts_loop, args=(args, config))

    t1.start()
    t2.start()

    # Block until vision loop ends or signal received
    while t1.is_alive():
        t1.join(timeout=0.5)
        if stop_event.is_set() and t2.is_alive():
            t2.join(timeout=1.0)

    if t2.is_alive():
        t2.join(timeout=2.0)

    log_event("COMPLETED", "Program stopped gracefully")


if __name__ == "__main__":
    main()