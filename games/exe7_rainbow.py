#!/usr/bin/env python3
"""
Game: Overhead Rainbow Toss (Exercise 7)
Description: YOLO detection of person + ball with quadrant-based sequence validation for rainbow toss exercise.
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
# QUADRANT CONSTANTS
# =============================================================================

QUADRANT_TOP_RIGHT = 0
QUADRANT_TOP_LEFT = 1
QUADRANT_BOTTOM_LEFT = 2
QUADRANT_BOTTOM_RIGHT = 3

QUADRANT_NAMES = ["TOP-RIGHT", "TOP-LEFT", "BOTTOM-LEFT", "BOTTOM-RIGHT"]


# =============================================================================
# MAIN GAME LOGIC
# =============================================================================

def vision_loop(args, config):
    """YOLO person + ball detection with quadrant sequence validation."""
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
    ball_label = config.get("ball_label", "ball")
    person_label = config.get("person_label", "person")
    ball_conf_threshold = config.get("ball_conf_threshold", 0.75)
    person_conf_threshold = config.get("person_conf_threshold", 0.5)
    frame_skip = config.get("frame_skip", 2)

    sets = config.get("sets", 2)
    reps_per_set = config.get("reps_per_set", 8)
    rest_time = config.get("rest_time_seconds", 30)
    min_time_between_quadrants = config.get("min_time_between_quadrants", 0.5)

    # State
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    current_set = 0
    current_rep = 0
    reps_completed = 0
    total_reps = 0

    ball_sequence = []
    rep_started = False
    rep_completed = False

    last_quadrant_change = 0

    game_active = False
    setup_mode = not args.headless  # Auto-skip setup in headless

    if args.headless:
        game_active = True
        current_set, current_rep, reps_completed = start_next_set(1, tts_queue)

    data_buffer = []
    frame_counter = 0
    prev_time = time.time()
    fps = 0

    log_event("READY", "Overhead Rainbow Toss active", {
        "sets": sets,
        "reps_per_set": reps_per_set,
        "frame_skip": frame_skip,
        "setup_mode": setup_mode
    })

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                log_event("WARN", "No frame detected, ending loop")
                break

            frame_counter += 1

            # --------------------------
            # SETUP MODE
            # --------------------------
            if setup_mode and not game_active:
                draw_setup_guide(frame, frame_h, frame_w)
                cv2.imshow(args.game_id, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    setup_mode = False
                    game_active = True
                    tts_queue.put("Setup complete. Get ready for Rainbow Toss!")
                    time.sleep(2)
                    current_set, current_rep, reps_completed = start_next_set(1, tts_queue)
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
            # YOLO DETECTION
            # --------------------------
            person_detected = False
            person_center_x = 0
            person_center_y = 0
            ball_center = None

            if frame_counter % frame_skip == 0:
                try:
                    results = model(frame, verbose=False)
                except Exception as e:
                    log_event("ERROR", f"YOLO inference error: {e}")
                    stop_event.set()
                    break

                for r in results:
                    if r.boxes is None:
                        continue
                    boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                    clss = r.boxes.cls.cpu().numpy().astype(int)
                    confidences = r.boxes.conf.cpu().numpy()

                    for box, cls, conf in zip(boxes, clss, confidences):
                        label = model.names[int(cls)] if int(cls) < len(model.names) else "unknown"
                        x1, y1, x2, y2 = box
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        if label == person_label and conf > person_conf_threshold:
                            person_center_x = cx
                            person_center_y = cy
                            person_detected = True

                            if not args.headless:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 5),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                        elif label == ball_label and conf > ball_conf_threshold:
                            ball_center = (cx, cy)
                            if not args.headless:
                                cv2.circle(frame, ball_center, 10, (0, 255, 255), -1)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"Ball {conf:.2f}", (x1, y1 - 5),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # --------------------------
            # QUADRANT LOGIC
            # --------------------------
            if person_detected:
                if not args.headless:
                    draw_quadrants(frame, person_center_x, person_center_y, frame_w, frame_h, ball_sequence)

                current_time = time.time()

                if ball_center:
                    ball_x, ball_y = ball_center
                    quadrant = get_ball_quadrant(ball_x, ball_y, person_center_x, person_center_y)

                    if not ball_sequence or (quadrant != ball_sequence[-1] and
                                            current_time - last_quadrant_change > min_time_between_quadrants):
                        ball_sequence.append(quadrant)
                        last_quadrant_change = current_time

                    if not args.headless:
                        cv2.putText(frame, f"Current: {QUADRANT_NAMES[quadrant]}",
                                   (person_center_x + 20, person_center_y + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Rep logic
                if game_active and not rep_completed:
                    if not rep_started and ball_center:
                        if ball_sequence and ball_sequence[-1] == QUADRANT_TOP_RIGHT:
                            rep_started = True
                            ball_sequence = [QUADRANT_TOP_RIGHT]
                            tts_queue.put("Rep started. Toss in rainbow arc!")

                    elif rep_started:
                        if is_valid_sequence(ball_sequence):
                            if len(ball_sequence) >= 3:
                                current_rep += 1
                                reps_completed += 1
                                total_reps += 1
                                rep_completed = True

                                tts_queue.put(f"Rep {current_rep} complete! Great rainbow arc.")

                                data_buffer.append({
                                    "timestamp": datetime.now().isoformat(),
                                    "event": "rep_complete",
                                    "set_number": current_set,
                                    "rep_number": current_rep,
                                    "sequence": ball_sequence.copy()
                                })

                                ball_sequence = []
                                rep_started = False
                                rep_completed = False

                                if reps_completed >= reps_per_set:
                                    if current_set < sets:
                                        tts_queue.put(f"Take {rest_time} seconds rest before set {current_set + 1}.")
                                        for _ in range(rest_time):
                                            if stop_event.is_set():
                                                break
                                            time.sleep(1)
                                        if not stop_event.is_set():
                                            current_set, current_rep, reps_completed = start_next_set(current_set + 1, tts_queue)
                                    else:
                                        tts_queue.put("All sets complete! Fantastic rainbow tosses!")
                                        game_active = False

                        # Invalid path: TOP-RIGHT → BOTTOM-RIGHT
                        if (ball_sequence and len(ball_sequence) >= 2 and
                            ball_sequence[0] == QUADRANT_TOP_RIGHT and
                            ball_sequence[-1] == QUADRANT_BOTTOM_RIGHT):
                            tts_queue.put("Invalid path. Try rainbow arc again.")
                            ball_sequence = []
                            rep_started = False

            else:
                if not args.headless:
                    cv2.putText(frame, "Person not detected", (30, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # --------------------------
            # VISUALS & STATUS
            # --------------------------
            if not args.headless:
                cv2.putText(frame, f"FPS: {int(fps)}", (frame_w - 120, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if game_active:
                    status_text = f"Set {current_set}/{sets} | Rep {current_rep}/{reps_per_set}"
                    cv2.putText(frame, status_text, (30, frame_h - 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    if not person_detected:
                        cv2.putText(frame, "Stand in view", (30, frame_h - 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif not ball_center:
                        cv2.putText(frame, "Show the ball", (30, frame_h - 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "Ready", (30, frame_h - 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    if not setup_mode:
                        cv2.putText(frame, "GAME COMPLETE", (frame_w // 2 - 100, frame_h // 2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

                cv2.imshow(args.game_id, frame)

                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    stop_event.set()
                    break
                elif key == ord('r'):
                    tts_queue.put("Resetting game.")
                    current_set = 0
                    current_rep = 0
                    reps_completed = 0
                    total_reps = 0
                    ball_sequence = []
                    rep_started = False
                    rep_completed = False
                    game_active = False
                    setup_mode = True

            # Frame metrics
            data_buffer.append({
                "timestamp": datetime.now().isoformat(),
                "frame": frame_counter,
                "set_number": current_set,
                "rep_number": current_rep,
                "person_detected": person_detected,
                "ball_detected": ball_center is not None,
                "ball_sequence": ball_sequence.copy() if ball_sequence else [],
                "game_active": game_active,
                "fps": fps
            })

    except Exception as e:
        log_event("ERROR", f"Vision loop exception: {str(e)}")
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tts_queue.put("__STOP__")
        log_event("CLEANUP", "Camera released", {"total_reps": total_reps})
        save_data(data_buffer, args.output, args.game_id, "rainbow")

    log_event("COMPLETED", "Overhead Rainbow Toss ended", {"total_reps": total_reps})


# =============================================================================
# QUADRANT FUNCTIONS
# =============================================================================

def get_ball_quadrant(ball_x, ball_y, person_x, person_y):
    """Determine which quadrant the ball is in (relative to person)."""
    if ball_y < person_y:  # Top half
        return QUADRANT_TOP_RIGHT if ball_x > person_x else QUADRANT_TOP_LEFT
    else:  # Bottom half
        return QUADRANT_BOTTOM_LEFT if ball_x < person_x else QUADRANT_BOTTOM_RIGHT


def is_valid_sequence(sequence):
    """Validate the ball sequence for one rep."""
    if len(sequence) < 3:
        return False
    if sequence[0] != QUADRANT_TOP_RIGHT:
        return False
    if sequence[1] != QUADRANT_TOP_LEFT:
        return False
    if sequence[2] != QUADRANT_BOTTOM_LEFT:
        return False
    return True


def draw_quadrants(frame, person_x, person_y, frame_w, frame_h, ball_sequence):
    """Draw 4 quadrants centered on person with highlight for next target."""
    cv2.line(frame, (0, person_y), (frame_w, person_y), (200, 200, 200), 2)
    cv2.line(frame, (person_x, 0), (person_x, frame_h), (200, 200, 200), 2)

    cv2.putText(frame, "TOP-RIGHT", (person_x + 20, person_y - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, "TOP-LEFT", (20, person_y - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, "BOTTOM-LEFT", (20, person_y + 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, "BOTTOM-RIGHT", (person_x + 20, person_y + 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # Highlight current requirement
    if not ball_sequence:
        overlay = frame.copy()
        cv2.rectangle(overlay, (person_x, 0), (frame_w, person_y), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.rectangle(frame, (person_x, 0), (frame_w, person_y), (0, 255, 0), 3)
        cv2.putText(frame, "START HERE →", (person_x + 50, person_y - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    elif len(ball_sequence) == 1 and ball_sequence[0] == QUADRANT_TOP_RIGHT:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (person_x, person_y), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.rectangle(frame, (0, 0), (person_x, person_y), (0, 255, 0), 3)
        cv2.putText(frame, "→ NEXT: TOP-LEFT", (50, person_y - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    elif (len(ball_sequence) >= 2 and
          ball_sequence[:2] == [QUADRANT_TOP_RIGHT, QUADRANT_TOP_LEFT]):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, person_y), (person_x, frame_h), (255, 255, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.rectangle(frame, (0, person_y), (person_x, frame_h), (255, 255, 0), 3)
        cv2.putText(frame, "→ THEN: BOTTOM-LEFT", (50, person_y + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


def draw_setup_guide(frame, frame_h, frame_w):
    """Draw camera setup instructions."""
    cv2.putText(frame, "OVERHEAD RAINBOW TOSS SETUP", (30, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(frame, "1. Stand 6-8 feet from camera", (30, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "2. Camera at shoulder height", (30, 110),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "3. Frame: full body visible", (30, 140),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "4. Ball should be clearly visible", (30, 170),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, "Press 's' when ready to start", (30, frame_h - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def start_next_set(set_num, tts_queue):
    """Start a new set."""
    tts_queue.put(f"Starting set {set_num}. Ready for rainbow toss.")
    return set_num, 0, 0


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
        description="Overhead Rainbow Toss with YOLO Quadrant Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python games/exe7_rainbow.py --game-id exe7 --camera 0
  python games/exe7_rainbow.py --game-id exe7 --camera "/path/to/video.mp4" --config configs/exe7_rainbow.json --headless
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