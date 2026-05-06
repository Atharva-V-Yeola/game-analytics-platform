#!/usr/bin/env python3
"""
Game: Ball Dash & Drop (Exercise 8)
Description: YOLO person + ball detection with phase-based relay: start → run to bucket → drop ball → run back.
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
# GAME PHASES
# =============================================================================

WAITING = 0
RUNNING_TO_BUCKET = 1
AT_BUCKET = 2
RUNNING_BACK = 3

PHASE_NAMES = ["WAITING", "RUNNING_TO_BUCKET", "AT_BUCKET", "RUNNING_BACK"]


# =============================================================================
# MAIN GAME LOGIC
# =============================================================================

def vision_loop(args, config):
    """YOLO detection with phase-based relay logic."""
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
    person_label = config.get("person_label", "person")
    ball_label = config.get("ball_label", "ball")
    conf_threshold = config.get("conf_threshold", 0.5)
    use_tracking = config.get("use_tracking", False)
    tracker = config.get("tracker", "bytetrack.yaml")
    device = config.get("device", "cpu")
    imgsz = config.get("inference_size", 640)

    sets = config.get("sets", 2)
    reps_per_set = config.get("reps_per_set", 1)
    rest_time = config.get("rest_time_seconds", 60)
    distance_meters = config.get("distance_meters", 15)

    start_line_ratio = config.get("start_line_ratio", 0.2)
    end_line_ratio = config.get("end_line_ratio", 0.8)
    tolerance = config.get("position_tolerance", 50)

    # State
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    current_set = 1
    current_rep = 0
    phase = WAITING

    reps_completed = 0
    reps_ball_dropped = 0
    ball_seen_initially = False

    data_buffer = []
    frame_counter = 0

    log_event("READY", "Ball Dash & Drop active", {
        "sets": sets,
        "reps_per_set": reps_per_set,
        "distance_meters": distance_meters,
        "rest_time": rest_time
    })

    tts_queue.put("Ball Dash and Drop. Stand at green line with ball.")

    try:
        while not stop_event.is_set() and current_set <= sets:
            ret, frame = cap.read()
            if not ret:
                log_event("WARN", "No frame detected, ending loop")
                break

            frame_counter += 1
            h, w = frame.shape[:2]

            start_line = int(w * start_line_ratio)
            end_line = int(w * end_line_ratio)

            # YOLO detection
            try:
                if use_tracking:
                    results = model.track(
                        frame,
                        persist=True,
                        tracker=tracker,
                        verbose=False,
                        conf=conf_threshold,
                        device=device,
                        imgsz=imgsz
                    )
                else:
                    results = model(
                        frame,
                        verbose=False,
                        conf=conf_threshold,
                        device=device,
                        imgsz=imgsz
                    )
            except Exception as e:
                log_event("ERROR", f"YOLO inference error: {e}")
                stop_event.set()
                break

            person_x, person_box, ball_visible, ball_boxes = get_person_and_ball(
                results, model, person_label, ball_label
            )

            # Draw visuals
            if not args.headless:
                draw_game(frame, phase, start_line, end_line, w, h)
                draw_boxes(frame, person_box, ball_boxes)
                draw_info(frame, current_set, current_rep, ball_visible, reps_completed, reps_ball_dropped, sets, reps_per_set)

            # Phase logic
            if phase == WAITING:
                if person_x and ball_visible:
                    if abs(person_x - start_line) < tolerance:
                        current_rep += 1
                        phase = RUNNING_TO_BUCKET
                        ball_seen_initially = True
                        log_event("REP_START", f"Set {current_set} Rep {current_rep} - GO!")
                        tts_queue.put("Go!")

            elif phase == RUNNING_TO_BUCKET:
                if person_x and person_x > end_line:
                    phase = AT_BUCKET
                    log_event("AT_BUCKET", "At bucket - drop ball!")
                    tts_queue.put("Drop the ball!")

            elif phase == AT_BUCKET:
                if ball_seen_initially and not ball_visible:
                    reps_ball_dropped += 1
                    phase = RUNNING_BACK
                    log_event("BALL_DROPPED", "Ball dropped successfully")
                    tts_queue.put("Run back!")
                elif person_x and person_x < end_line - tolerance:
                    phase = RUNNING_BACK
                    log_event("LEFT_BUCKET", "Left bucket without dropping")
                    tts_queue.put("Run back!")

            elif phase == RUNNING_BACK:
                if person_x and abs(person_x - start_line) < tolerance:
                    reps_completed += 1
                    current_rep += 1
                    log_event("REP_COMPLETE", f"Rep {current_rep} complete!", {
                        "ball_dropped": ball_seen_initially and not ball_visible
                    })

                    if current_rep >= reps_per_set:
                        log_event("SET_COMPLETE", f"Set {current_set} complete!", {
                            "reps_completed": reps_completed,
                            "reps_ball_dropped": reps_ball_dropped
                        })
                        tts_queue.put(f"Set {current_set} complete!")

                        if current_set < sets:
                            # Rest period
                            tts_queue.put(f"Rest {rest_time} seconds")
                            if not args.headless:
                                rest_countdown(cap, rest_time, w, h, args.game_id)
                            else:
                                for _ in range(rest_time):
                                    if stop_event.is_set():
                                        break
                                    time.sleep(1)

                            if not stop_event.is_set():
                                current_set += 1
                                current_rep = 0
                                tts_queue.put(f"Set {current_set} - start!")
                        else:
                            tts_queue.put("All sets complete! Great job!")
                            log_event("GAME_COMPLETE", "All sets finished", {
                                "total_reps": reps_completed,
                                "total_ball_dropped": reps_ball_dropped
                            })
                            break

                        phase = WAITING
                        ball_seen_initially = False
                    else:
                        phase = WAITING
                        ball_seen_initially = False

            # Frame metrics
            data_buffer.append({
                "timestamp": datetime.now().isoformat(),
                "frame": frame_counter,
                "set_number": current_set,
                "rep_number": current_rep,
                "phase": PHASE_NAMES[phase],
                "person_detected": person_x is not None,
                "ball_visible": ball_visible,
                "ball_seen_initially": ball_seen_initially,
                "reps_completed": reps_completed,
                "reps_ball_dropped": reps_ball_dropped
            })

            if not args.headless:
                cv2.imshow(args.game_id, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    log_event("STOP", "User pressed 'q'")
                    stop_event.set()
                    break

    except Exception as e:
        log_event("ERROR", f"Vision loop exception: {str(e)}")
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tts_queue.put("__STOP__")
        log_event("CLEANUP", "Camera released", {
            "final_set": current_set,
            "total_reps": reps_completed,
            "total_ball_dropped": reps_ball_dropped
        })
        save_data(data_buffer, args.output, args.game_id, "dashdrop")

    log_event("COMPLETED", "Ball Dash & Drop ended", {
        "total_reps": reps_completed,
        "total_ball_dropped": reps_ball_dropped
    })


# =============================================================================
# DETECTION & DRAWING FUNCTIONS
# =============================================================================

def get_person_and_ball(results, model, person_label, ball_label):
    """Extract person and ball detections from YOLO results."""
    person_x = None
    person_box = None
    ball_visible = False
    ball_boxes = []

    for r in results:
        if r.boxes is None:
            continue
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        classes = r.boxes.cls.cpu().numpy().astype(int)

        for box, cls in zip(boxes, classes):
            name = model.names[int(cls)] if int(cls) < len(model.names) else "unknown"

            if name == person_label:
                x1, y1, x2, y2 = box
                person_x = (x1 + x2) // 2
                person_box = box
            elif name == ball_label:
                ball_visible = True
                ball_boxes.append(box)

    return person_x, person_box, ball_visible, ball_boxes


def draw_game(frame, phase, start_line, end_line, w, h):
    """Draw game lines and phase-based text."""
    cv2.line(frame, (start_line, 0), (start_line, h), (0, 255, 0), 4)
    cv2.line(frame, (end_line, 0), (end_line, h), (255, 0, 0), 4)

    if phase == WAITING:
        cv2.putText(frame, "START HERE", (start_line - 80, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    elif phase == RUNNING_TO_BUCKET:
        cv2.putText(frame, "RUN TO BUCKET >>", (w // 2 - 150, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
    elif phase == AT_BUCKET:
        cv2.putText(frame, "DROP BALL!", (end_line - 100, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
    elif phase == RUNNING_BACK:
        cv2.putText(frame, "<< RUN BACK", (w // 2 - 100, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)


def draw_boxes(frame, person_box, ball_boxes):
    """Draw bounding boxes for person and balls."""
    if person_box is not None:
        x1, y1, x2, y2 = person_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 3)
        cv2.putText(frame, "PLAYER", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

    for ball_box in ball_boxes:
        x1, y1, x2, y2 = ball_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, "BALL", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def draw_info(frame, current_set, current_rep, ball_visible, reps_completed, reps_ball_dropped, sets, reps_per_set):
    """Draw info box with game status."""
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (10, h - 180), (350, h - 10), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, h - 180), (350, h - 10), (255, 255, 255), 2)

    y = h - 150
    cv2.putText(frame, f"Set: {current_set}/{sets}", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Rep: {current_rep}/{reps_per_set}", (20, y + 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    ball_color = (0, 255, 0) if ball_visible else (0, 0, 255)
    cv2.putText(frame, f"Ball: {'YES' if ball_visible else 'NO'}", (20, y + 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, ball_color, 2)

    cv2.putText(frame, f"Completed: {reps_completed}", (20, y + 105),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Ball Dropped: {reps_ball_dropped}", (20, y + 135),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def rest_countdown(cap, rest_time, w, h, window_name):
    """Show rest countdown with live video feed."""
    for countdown in range(rest_time, 0, -1):
        if stop_event.is_set():
            break
        ret, rest_frame = cap.read()
        if ret:
            cv2.putText(rest_frame, f"REST: {countdown}s",
                       (w // 2 - 100, h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 3)
            cv2.imshow(window_name, rest_frame)
            
            # Use short delays to remain responsive to shutdown signals
            for _ in range(10):
                if stop_event.is_set():
                    break
                cv2.waitKey(100)


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
        description="Ball Dash & Drop - YOLO Phase-Based Relay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python games/exe8_dashdrop.py --game-id exe8 --camera 0
  python games/exe8_dashdrop.py --game-id exe8 --camera "/path/to/video.mp4" --config configs/exe8_dashdrop.json --headless
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

    # Run vision loop in main thread
    vision_loop(args, config)

    # Wait for TTS to finish
    tts_thread.join(timeout=10.0)

    log_event("COMPLETED", "Program stopped gracefully")


if __name__ == "__main__":
    main()