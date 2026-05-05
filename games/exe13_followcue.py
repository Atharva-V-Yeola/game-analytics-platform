#!/usr/bin/env python3
"""
Game: Follow-the-Cue Pass (Exercise 13)
Description: Hash-based PRNG target color, YOLO ball tracking across 4 zones.
"""

import argparse
import cv2
import hashlib
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
# CONSTANTS & UTILS
# =============================================================================

COLORS = {
    0: ("Red",    (0,   0,   220)),
    1: ("Blue",   (220, 0,   0  )),
    2: ("Green",  (0,   200, 0  )),
    3: ("Yellow", (0,   220, 220)),
}

def get_color_index(seed: int, counter: int) -> int:
    """Deterministic pseudo-random color via SHA-256."""
    key = f"{seed}_{counter}".encode()
    digest = hashlib.sha256(key).hexdigest()
    return int(digest, 16) % 4

def get_zone_regions(frame_w, frame_h):
    """Right half is split into 4 equal vertical strips."""
    start_x  = frame_w // 2
    strip_w  = (frame_w - start_x) // 4
    zones = []
    for i in range(4):
        x1 = start_x + i * strip_w
        x2 = x1 + strip_w
        zones.append((x1, x2))
    return zones

def ball_zone(cx: int, zones: list) -> int:
    for i, (x1, x2) in enumerate(zones):
        if x1 <= cx < x2:
            return i
    return -1


# =============================================================================
# SHARED STATE
# =============================================================================

class SharedState:
    def __init__(self):
        self.lock          = threading.Lock()
        self.target_color  = -1
        self.current_set   = 0
        self.correct_count = 0
        self.total_correct = 0
        self.pass_counter  = 0
        self.waiting_throw = False
        self.rest_mode     = False
        self.hit_status    = None # None, "CORRECT", "WRONG"
        self.hit_time      = 0


# =============================================================================
# GAME LOGIC THREAD
# =============================================================================

def game_loop(state, config):
    sets = config.get("sets", 4)
    passes = config.get("passes_per_set", 6)
    rest_seconds = config.get("rest_seconds", 30)
    seed = config.get("seed", 42)

    tts_queue.put("Exercise starting. Get ready.")
    time.sleep(2)

    for set_num in range(1, sets + 1):
        if stop_event.is_set():
            return

        with state.lock:
            state.current_set   = set_num
            state.correct_count = 0
            state.rest_mode     = False
            state.hit_status    = None

        tts_queue.put(f"Set {set_num}. Begin.")
        log_event("SET_START", f"Starting set {set_num}")
        time.sleep(2)

        for pass_num in range(1, passes + 1):
            if stop_event.is_set():
                return

            with state.lock:
                idx = get_color_index(seed, state.pass_counter)
                state.target_color = idx
                state.waiting_throw = True
                state.hit_status = None
                state.pass_counter += 1

            color_name = COLORS[idx][0]
            tts_queue.put(f"Pass {pass_num}. {color_name}.")
            log_event("TARGET", f"Target color: {color_name}")

            # Wait for throw
            timeout = 8.0
            t0 = time.time()
            while time.time() - t0 < timeout:
                if stop_event.is_set():
                    return
                with state.lock:
                    if not state.waiting_throw:
                        break
                time.sleep(0.05)

            time.sleep(0.4)

        with state.lock:
            correct = state.correct_count
            state.total_correct += correct
            state.rest_mode = True
            state.target_color = -1
            state.waiting_throw = False
            state.hit_status = None

        tts_queue.put(f"Set {set_num} complete. {correct} correct out of {passes}.")
        log_event("SET_COMPLETE", f"Set {set_num} finished", {"correct": correct})
        time.sleep(3)

        if set_num < sets:
            tts_queue.put(f"Rest for {rest_seconds} seconds.")
            for remaining in range(rest_seconds, 0, -5):
                if stop_event.is_set():
                    return
                if remaining <= 10:
                    tts_queue.put(f"{remaining} seconds.")
                time.sleep(5)
            tts_queue.put("Get ready for the next set.")
            time.sleep(2)

    with state.lock:
        total = state.total_correct
    tts_queue.put(f"Exercise complete! You scored {total} correct passes out of {sets * passes}. Great work!")
    log_event("GAME_COMPLETE", "All sets completed", {"total_correct": total})


# =============================================================================
# VISION LOOP
# =============================================================================

def vision_loop(args, config, state):
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

    tracker = config.get("tracker", "bytetrack.yaml")
    cross_line_x = config.get("cross_line_x", 320)
    
    seen_ids = set()
    data_buffer = []
    frame_counter = 0

    log_event("READY", "Follow-the-Cue Pass active", {
        "tracker": tracker,
        "cross_line_x": cross_line_x
    })

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                log_event("WARN", "No frame detected, ending loop")
                break
                
            frame_counter += 1
            frame_h, frame_w = frame.shape[:2]
            
            # dynamically adjust cross line if config provides a hardcoded one that exceeds w
            actual_cross_line = min(cross_line_x, frame_w // 2 - 10) if frame_w > 0 else cross_line_x
            zones = get_zone_regions(frame_w, frame_h)

            try:
                results = model.track(frame, persist=True, tracker=tracker, verbose=False)
            except Exception as e:
                log_event("ERROR", f"YOLO inference error: {e}")
                break

            with state.lock:
                target_idx    = state.target_color
                waiting       = state.waiting_throw
                rest          = state.rest_mode
                current_set   = state.current_set
                correct_count = state.correct_count
                total_correct = state.total_correct
                hit_status    = state.hit_status
                hit_time      = state.hit_time
                
                # Clear visual status after 2 seconds
                if hit_status and time.time() - hit_time > 2.0:
                    state.hit_status = None
                    hit_status = None

            for r in results:
                if r.boxes is None:
                    continue
                boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                clss  = r.boxes.cls.cpu().numpy().astype(int)
                ids   = (r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [])

                for (x1, y1, x2, y2), cls, oid in zip(boxes, clss, ids):
                    label = model.names[cls] if cls < len(model.names) else "unknown"
                    if label != "ball":
                        continue

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    if not args.headless:
                        cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 2)
                        cv2.putText(frame, f"ID:{oid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                    if cx > actual_cross_line and oid not in seen_ids and waiting:
                        hit_zone = ball_zone(cx, zones)
                        seen_ids.add(oid)

                        if hit_zone == target_idx and target_idx >= 0:
                            with state.lock:
                                state.correct_count += 1
                                state.waiting_throw  = False
                                state.hit_status     = "CORRECT"
                                state.hit_time       = time.time()
                            log_event("PASS", "Correct zone hit", {"zone": COLORS[target_idx][0]})
                        elif hit_zone != -1 and target_idx >= 0:
                            with state.lock:
                                state.waiting_throw  = False
                                state.hit_status     = "WRONG"
                                state.hit_time       = time.time()
                            log_event("PASS", "Wrong zone hit", {"expected": COLORS[target_idx][0], "got": COLORS[hit_zone][0]})

            with state.lock:
                if not state.waiting_throw:
                    seen_ids.clear()

            data_buffer.append({
                "timestamp": datetime.now().isoformat(),
                "frame": frame_counter,
                "set": current_set,
                "target_color": COLORS[target_idx][0] if target_idx >= 0 else None,
                "correct_count": correct_count,
                "total_correct": total_correct,
                "waiting": waiting,
                "rest": rest
            })

            if not args.headless:
                # Draw Zones
                for i, (x1, x2) in enumerate(zones):
                    name, bgr = COLORS[i]
                    alpha = 0.35 if i == target_idx else 0.15
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, 0), (x2, frame_h), bgr, -1)
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                    thickness = 4 if i == target_idx else 1
                    cv2.rectangle(frame, (x1, 0), (x2, frame_h), bgr, thickness)
                    cv2.putText(frame, name, (x1 + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)

                # Cross Line
                cv2.line(frame, (actual_cross_line, 0), (actual_cross_line, frame_h), (200, 200, 200), 2)

                if rest:
                    hud = "REST"
                elif target_idx >= 0:
                    hud = f"TARGET -> {COLORS[target_idx][0]}"
                else:
                    hud = "WAITING..."

                cv2.putText(frame, hud, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 0), 3)
                
                sets_total = config.get("sets", 4)
                passes_total = config.get("passes_per_set", 6)
                
                cv2.putText(frame, f"Set: {current_set}/{sets_total}  Correct: {correct_count}/{passes_total}",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
                cv2.putText(frame, f"Total Correct: {total_correct}",
                            (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                
                if hit_status == "CORRECT":
                    cv2.putText(frame, "CORRECT!", (50, frame_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
                elif hit_status == "WRONG":
                    cv2.putText(frame, "WRONG ZONE", (50, frame_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

                cv2.imshow(args.game_id, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break
            else:
                time.sleep(0.01)

    except Exception as e:
        log_event("ERROR", f"Vision loop exception: {str(e)}")
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()
        save_data(data_buffer, args.output, args.game_id, "followcue")
        log_event("CLEANUP", "Camera released")


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
    parser = argparse.ArgumentParser(description="Follow-the-Cue Pass")
    parser.add_argument("--game-id", required=True, help="Unique game identifier")
    parser.add_argument("--camera", default="0", help="Camera index or video file")
    parser.add_argument("--output", default="./data", help="CSV output directory")
    parser.add_argument("--config", default=None, help="Path to JSON config")
    parser.add_argument("--model-path", default="models/all_in_one_yolov82/weights/best.pt", help="YOLO model")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")

    args = parser.parse_args()
    config = load_config(args.config)

    log_event("INIT", "Game starting", {"game_id": args.game_id})

    state = SharedState()

    t_tts = threading.Thread(target=tts_worker, args=(config,))
    t_game = threading.Thread(target=game_loop, args=(state, config))

    t_tts.start()
    t_game.start()

    vision_loop(args, config, state)

    t_game.join(timeout=2.0)
    tts_queue.put("__STOP__")
    t_tts.join(timeout=5.0)

    log_event("COMPLETED", "Program stopped gracefully")


if __name__ == "__main__":
    main()
