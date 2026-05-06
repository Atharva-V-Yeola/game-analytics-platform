#!/usr/bin/env python3
"""
Game: Balance Statue (Exercise 2)
Description: MediaPipe pose-based single-leg balance hold with TTS coaching.
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

# Optional imports with graceful fallback
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


# =============================================================================
# GLOBALS & SIGNAL HANDLING
# =============================================================================

stop_event = threading.Event()
tts_queue = queue.Queue()  # Thread-safe queue for TTS messages

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
# TTS THREAD (Non-blocking)
# =============================================================================

def tts_worker(config):
    """
    CHANGE: TTS runs in dedicated thread so vision loop never blocks.
    Vision loop puts messages in tts_queue; this thread speaks them.
    """
    if not TTS_AVAILABLE:
        log_event("INFO", "TTS not available")
        return

    engine = pyttsx3.init()
    engine.setProperty('rate', config.get("tts_rate", 160))
    engine.setProperty('volume', config.get("tts_volume", 1.0))

    # CHANGE: Speak initial message
    engine.say("Balance game started. Get ready on your left leg.")
    engine.runAndWait()

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
    """
    CHANGE: All vision processing happens here.
    TTS is decoupled via tts_queue.
    """
    if not MEDIAPIPE_AVAILABLE:
        log_event("ERROR", "mediapipe not installed")
        return

    # Load config parameters (CHANGE: extracted from hardcoded values)
    sets_per_leg = config.get("sets_per_leg", 3)
    hold_time = config.get("hold_time_seconds", 20)
    rest_time = config.get("rest_time_seconds", 30)
    threshold = config.get("balance_threshold", 0.05)
    detection_conf = config.get("min_detection_confidence", 0.5)
    tracking_conf = config.get("min_tracking_confidence", 0.5)

    # State variables
    current_leg = config.get("starting_leg", "LEFT")
    sets_done = 0
    hold_start = None
    rest_start = None
    in_rest = False
    game_over = False

    # Data collection (CHANGE: added for analytics)
    data_buffer = []
    frame_counter = 0

    # MediaPipe setup
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Open camera
    cap = open_camera(args.camera)
    if not cap.isOpened():
        log_event("ERROR", "Cannot open camera", {"camera": args.camera})
        sys.exit(1)

    log_event("READY", "Balance game active", {
        "sets_per_leg": sets_per_leg,
        "hold_time": hold_time,
        "rest_time": rest_time,
        "threshold": threshold
    })

    try:
        with mp_pose.Pose(
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        ) as pose:

            while not stop_event.is_set() and not game_over:
                ret, frame = cap.read()
                if not ret:
                    log_event("ERROR", "Frame capture failed")
                    break

                frame_counter += 1
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                display = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Metrics for this frame
                frame_metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "frame": frame_counter,
                    "sets_done": sets_done,
                    "in_rest": in_rest,
                    "leg": current_leg
                }

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        display, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )
                    landmarks = results.pose_landmarks.landmark
                    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

                    diff = abs(left_ankle.y - right_ankle.y)
                    frame_metrics["ankle_diff"] = float(diff)

                    if not in_rest:
                        if diff > threshold:  # Standing on one leg
                            if hold_start is None:
                                hold_start = time.time()
                                tts_queue.put("Hold steady")  # CHANGE: Non-blocking TTS

                            elapsed = time.time() - hold_start
                            remaining = int(hold_time - elapsed)
                            frame_metrics["hold_remaining"] = remaining

                            cv2.putText(display, f"Holding... {max(0, remaining)} sec left",
                                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                            if elapsed >= hold_time:
                                sets_done += 1
                                tts_queue.put(f"Set {sets_done} complete. Take rest for {rest_time} seconds.")
                                in_rest = True
                                rest_start = time.time()
                                hold_start = None

                                if sets_done >= sets_per_leg:
                                    tts_queue.put("All sets completed. Game over. Great job!")
                                    game_over = True
                        else:
                            hold_start = None
                            frame_metrics["hold_remaining"] = None
                            cv2.putText(display, "Both legs down - reset!", (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        elapsed_rest = time.time() - rest_start
                        remaining_rest = int(rest_time - elapsed_rest)
                        frame_metrics["rest_remaining"] = remaining_rest

                        cv2.putText(display, f"Resting... {max(0, remaining_rest)} sec left",
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                        if elapsed_rest >= rest_time:
                            in_rest = False
                            tts_queue.put("Start next set. Balance again.")

                else:
                    frame_metrics["ankle_diff"] = None
                    frame_metrics["hold_remaining"] = None

                data_buffer.append(frame_metrics)

                if not args.headless:
                    cv2.imshow(args.game_id, display)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        log_event("STOP", "User pressed 'q'")
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
        tts_queue.put("__STOP__")  # Signal TTS thread to exit
        log_event("CLEANUP", "Camera released", {"final_sets": sets_done})
        save_data(data_buffer, args.output, args.game_id, "balance")

    log_event("COMPLETED", "Balance game ended", {"total_sets": sets_done})


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
    """Flush collected metrics to CSV."""
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
        description="Balance Statue Game with MediaPipe Pose",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python games/exe2_balance.py --game-id exe2 --camera 0
  python games/exe2_balance.py --game-id exe2 --camera 0 --config configs/exe2_balance.json --headless
        """
    )
    parser.add_argument("--game-id", required=True, help="Unique game identifier")
    parser.add_argument("--camera", default="0", help="Camera index or video file path")
    parser.add_argument("--output", default="./data", help="CSV output directory")
    parser.add_argument("--config", default=None, help="Path to JSON config file")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI display (disables TTS and cv2.imshow)")

    args = parser.parse_args()
    config = load_config(args.config)

    cv2.setNumThreads(2)

    log_event("INIT", "Game starting", {
        "game_id": args.game_id,
        "camera": args.camera,
        "platform": platform.system(),
        "python": sys.executable
    })

    # CHANGE: Launch TTS thread first
    tts_thread = threading.Thread(target=tts_worker, args=(config,))
    tts_thread.start()

    # CHANGE: Run vision loop in main thread
    vision_loop(args, config)

    # Wait for TTS to finish speaking remaining messages
    tts_thread.join(timeout=10.0)

    log_event("COMPLETED", "Program stopped gracefully")


if __name__ == "__main__":
    main()