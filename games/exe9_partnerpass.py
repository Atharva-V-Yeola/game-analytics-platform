#!/usr/bin/env python3
"""
Game: Partner Colour Pass (Exercise 9)
Description: De Bruijn sequence guided ball passing (Red=Chest/Vertical, Blue=Bounce/Horizontal)
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
# DE BRUIJN SEQUENCE
# =============================================================================

def generate_debruijn(k, n):
    """
    Generate De Bruijn sequence B(k,n)
    k = alphabet size, n = subsequence length
    """
    alphabet = ['Red', 'Blue']
    a = [0] * k * n
    sequence = []

    def db(t, p):
        if t > n:
            if n % p == 0:
                sequence.extend([a[i] for i in range(1, p + 1)])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    return [alphabet[i] for i in sequence]


# =============================================================================
# GAME LOGIC THREAD
# =============================================================================

def game_loop(game_state, lock, debruijn_colors, config):
    total_sets = config.get("total_sets", 3)
    passes_per_set = config.get("passes_per_set", 10)
    rest_time = config.get("rest_time_seconds", 60)

    tts_queue.put("Welcome to Partner Colour Pass Game. Get ready!")
    time.sleep(2)

    for set_num in range(1, total_sets + 1):
        if stop_event.is_set():
            break

        with lock:
            game_state['current_set'] = set_num
            game_state['count'] = 0
            game_state['sequence_index'] = 0
            game_state['break_case'] = False

        tts_queue.put(f"Set {set_num} starting")
        log_event("SET_START", f"Starting set {set_num}")
        time.sleep(1)

        while True:
            if stop_event.is_set():
                return

            with lock:
                if game_state['break_case']:
                    tts_queue.put("Wrong technique. Restarting set.")
                    log_event("RESTART", f"Set {set_num} restarted due to break case")
                    game_state['count'] = 0
                    game_state['sequence_index'] = 0
                    game_state['break_case'] = False
                    time.sleep(2)
                    continue

                if game_state['count'] >= passes_per_set:
                    break

                color = debruijn_colors[game_state['sequence_index'] % len(debruijn_colors)]
                game_state['current_color'] = color
                game_state['detection_enabled'] = True
                game_state['sequence_index'] += 1

            tts_queue.put(color)
            log_event("COLOR_CALLED", f"Color announced: {color}")

            # Wait for pass completion
            while True:
                if stop_event.is_set():
                    return

                with lock:
                    if game_state['break_case']:
                        break
                    if time.time() >= game_state['freeze_until'] and game_state['freeze_until'] > 0:
                        game_state['freeze_until'] = 0
                        break
                time.sleep(0.1)

            time.sleep(0.5)

        if set_num < total_sets:
            with lock:
                game_state['rest_mode'] = True
                game_state['detection_enabled'] = False

            tts_queue.put(f"Set {set_num} complete. {rest_time} seconds rest.")
            log_event("REST", f"Set {set_num} completed. Resting...")

            for i in range(rest_time):
                if stop_event.is_set():
                    return
                time.sleep(1)

            with lock:
                game_state['rest_mode'] = False

    if not stop_event.is_set():
        tts_queue.put("Congratulations! You have successfully completed all three sets. Great job!")
        log_event("GAME_COMPLETE", "All sets completed!")


# =============================================================================
# VISION LOOP
# =============================================================================

def vision_loop(args, config, game_state, lock):
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
    freeze_duration = config.get("freeze_duration_seconds", 7)
    
    # State tracking
    seen_ids = {}
    data_buffer = []
    frame_counter = 0

    log_event("READY", "Partner Colour Pass active", {
        "tracker": tracker,
        "freeze_duration": freeze_duration
    })

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                log_event("WARN", "No frame detected, ending loop")
                break
                
            frame_counter += 1
            h, w = frame.shape[:2]

            # Dynamic line resolution fallback if config doesn't match resolution well
            vline_x = config.get("vertical_line_x", w // 2)
            hline_y = config.get("horizontal_line_y", h // 2)

            current_time = time.time()

            with lock:
                detection_active = (
                    game_state['detection_enabled'] and 
                    not game_state['rest_mode'] and
                    current_time >= game_state['freeze_until']
                )
                current_color = game_state['current_color']
                count = game_state['count']
                current_set = game_state['current_set']
                passes_per_set = config.get("passes_per_set", 10)
                rest_mode = game_state['rest_mode']
                freeze_until = game_state['freeze_until']

            try:
                results = model.track(frame, persist=True, tracker=tracker, verbose=False)
            except Exception as e:
                log_event("ERROR", f"YOLO inference error: {e}")
                break

            if detection_active and current_color:
                for r in results:
                    if r.boxes is None or r.boxes.id is None:
                        continue

                    boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                    clss = r.boxes.cls.cpu().numpy().astype(int)
                    ids = r.boxes.id.cpu().numpy().astype(int)

                    for (x1, y1, x2, y2), cls, oid in zip(boxes, clss, ids):
                        label = model.names[cls] if cls < len(model.names) else "unknown"

                        if label == "ball":
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2

                            if oid not in seen_ids:
                                seen_ids[oid] = {'last_x': cx, 'last_y': cy, 'crossed': False}
                                continue

                            last_x = seen_ids[oid]['last_x']
                            last_y = seen_ids[oid]['last_y']
                            already_counted = seen_ids[oid]['crossed']

                            if not already_counted:
                                if current_color == 'Red':
                                    if last_x < vline_x <= cx:
                                        with lock:
                                            game_state['count'] += 1
                                            game_state['freeze_until'] = time.time() + freeze_duration
                                            seen_ids[oid]['crossed'] = True
                                        log_event("PASS", "RED chest pass", {"count": game_state['count']})
                                    elif (last_y < hline_y <= cy) or (last_y > hline_y >= cy):
                                        with lock:
                                            game_state['break_case'] = True
                                        log_event("BREAK", "Wrong line crossed for RED")

                                elif current_color == 'Blue':
                                    if last_y < hline_y <= cy:
                                        with lock:
                                            game_state['count'] += 1
                                            game_state['freeze_until'] = time.time() + freeze_duration
                                            seen_ids[oid]['crossed'] = True
                                        log_event("PASS", "BLUE bounce pass", {"count": game_state['count']})
                                    elif (last_x < vline_x <= cx) or (last_x > vline_x >= cx):
                                        with lock:
                                            game_state['break_case'] = True
                                        log_event("BREAK", "Wrong line crossed for BLUE")

                            seen_ids[oid]['last_x'] = cx
                            seen_ids[oid]['last_y'] = cy
                            
                            if not args.headless:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            data_buffer.append({
                "timestamp": datetime.now().isoformat(),
                "frame": frame_counter,
                "set_number": current_set,
                "count": count,
                "target_color": current_color,
                "status": "REST" if rest_mode else ("FROZEN" if current_time < freeze_until else current_color)
            })

            if not args.headless:
                cv2.line(frame, (vline_x, 0), (vline_x, h), (0, 0, 255), 3)  # Red vertical
                cv2.line(frame, (0, hline_y), (w, hline_y), (255, 0, 0), 3)  # Blue horizontal

                status = "REST" if rest_mode else ("FROZEN" if current_time < freeze_until else current_color)
                
                cv2.putText(frame, f"Set: {current_set} | Count: {count}/{passes_per_set}", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Status: {status}", 
                           (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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
        save_data(data_buffer, args.output, args.game_id, "partnerpass")
        log_event("CLEANUP", "Camera released")


# =============================================================================
# UTILITIES
# =============================================================================

def open_camera(camera_id):
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
    parser = argparse.ArgumentParser(description="Partner Colour Pass")
    parser.add_argument("--game-id", required=True, help="Unique game identifier")
    parser.add_argument("--camera", default="0", help="Camera index or video file")
    parser.add_argument("--output", default="./data", help="CSV output directory")
    parser.add_argument("--config", default=None, help="Path to JSON config")
    parser.add_argument("--model-path", default="models/all_in_one_yolov82/weights/best.pt", help="YOLO model")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")

    args = parser.parse_args()
    config = load_config(args.config)

    log_event("INIT", "Game starting", {"game_id": args.game_id})

    # Initialize shared state
    game_state = {
        'current_set': 1,
        'count': 0,
        'sequence_index': 0,
        'current_color': None,
        'detection_enabled': False,
        'freeze_until': 0,
        'rest_mode': False,
        'break_case': False
    }
    lock = threading.Lock()
    
    # Generate sequence
    debruijn_colors = generate_debruijn(2, 4)

    # Start threads
    t_tts = threading.Thread(target=tts_worker, args=(config,))
    t_game = threading.Thread(target=game_loop, args=(game_state, lock, debruijn_colors, config))
    
    t_tts.start()
    t_game.start()

    # Run vision loop (blocking)
    vision_loop(args, config, game_state, lock)

    t_game.join(timeout=2.0)
    tts_queue.put("__STOP__")
    t_tts.join(timeout=5.0)

    log_event("COMPLETED", "Program stopped gracefully")


if __name__ == "__main__":
    main()
