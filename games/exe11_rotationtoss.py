#!/usr/bin/env python3
"""
Game: Rotational Toss (Exercise 11)
Description: MediaPipe pose + YOLO tracking for overhead rotational throw validation.
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

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


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
    """
    TTS runs in a dedicated thread.
    On Windows, it completely bypasses pyttsx3 and uses a PowerShell subprocess
    to prevent COM/SAPI5 deadlocks with MediaPipe.
    """
    if not TTS_AVAILABLE:
        log_event("INFO", "TTS not available")
        return

    is_windows = platform.system() == "Windows"
    engine = None

    # Only initialize pyttsx3 on Linux/Mac
    if not is_windows:
        engine = init_tts(config.get("tts_driver"))
        if engine:
            engine.setProperty('rate', config.get("tts_rate", 160))
            engine.setProperty('volume', config.get("tts_volume", 1.0))

    while not stop_event.is_set():
        try:
            msg = tts_queue.get(timeout=0.5)
            if msg == "__STOP__":
                break
            
            safe_msg = msg.replace("'", "")
            
            if is_windows:
                import subprocess
                cmd = f"Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate = 1; $s.Speak('{safe_msg}')"
                try:
                    subprocess.Popen(["powershell", "-Command", cmd], creationflags=0x08000000)
                except Exception as e:
                    log_event("ERROR", f"PowerShell TTS error: {e}")
            else:
                if engine:
                    engine.say(msg)
                    engine.runAndWait()
        except queue.Empty:
            continue
        except Exception as e:
            log_event("ERROR", f"TTS error: {e}")

    log_event("CLEANUP", "TTS thread ended")


# =============================================================================
# CONSTANTS
# =============================================================================
TOP_RIGHT = 0
TOP_LEFT = 1
BOTTOM_LEFT = 2
BOTTOM_RIGHT = 3


# =============================================================================
# MAIN GAME LOGIC
# =============================================================================

def vision_loop(args, config):
    if not ULTRALYTICS_AVAILABLE:
        log_event("ERROR", "ultralytics not installed")
        return
    if not MEDIAPIPE_AVAILABLE:
        log_event("ERROR", "mediapipe not installed")
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
    reps_per_set = config.get("reps_per_set", 10)
    pause_between_reps = config.get("pause_between_reps_seconds", 5)
    rest_between_sets = config.get("rest_between_sets_seconds", 60)
    yolo_conf_threshold = config.get("yolo_conf_threshold", 0.5)

    # State
    person_x, person_y = 0, 0
    person_detected = False
    head_y = 0
    ball_sequence = []
    rep_count = 0
    current_set = 1
    
    is_paused = False
    pause_start_time = 0
    is_resting = False
    rest_start_time = 0
    
    frame_counter = 0
    data_buffer = []

    log_event("READY", "Rotational Toss active", {
        "sets": sets,
        "reps_per_set": reps_per_set
    })

    tts_queue.put("Starting Rotational Toss Game. Position camera on your right side. Keep ball overhead and rotate from right to left before throwing.")

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    try:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    log_event("WARN", "No frame detected, ending loop")
                    break

                frame_counter += 1
                h, w = frame.shape[:2]
                current_time = time.time()

                # State handling: PAUSE
                if is_paused:
                    remaining = int(pause_between_reps - (current_time - pause_start_time))
                    if remaining > 0:
                        if not args.headless:
                            cv2.putText(frame, f"PAUSE: {remaining}s", (w//2-100, h//2), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 3)
                            cv2.putText(frame, f"Set: {current_set}/{sets} | Rep: {rep_count}/{reps_per_set}", 
                                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                            cv2.imshow(args.game_id, frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                stop_event.set()
                                break
                        else:
                            time.sleep(0.01)
                        continue
                    else:
                        is_paused = False
                        tts_queue.put("Resume")

                # State handling: REST
                if is_resting:
                    remaining = int(rest_between_sets - (current_time - rest_start_time))
                    if remaining > 0:
                        if remaining in [45, 30, 15, 5]:
                            # avoid queuing multiple times in a second
                            if getattr(vision_loop, "last_rest_announcement", None) != remaining:
                                tts_queue.put(f"{remaining} seconds remaining")
                                vision_loop.last_rest_announcement = remaining
                                
                        if not args.headless:
                            cv2.putText(frame, f"REST: {remaining}s", (w//2-150, h//2), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 165, 255), 4)
                            cv2.putText(frame, f"Set {current_set-1} Complete!", (w//2-200, h//2-80), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                            cv2.imshow(args.game_id, frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                stop_event.set()
                                break
                        else:
                            time.sleep(0.01)
                        continue
                    else:
                        is_resting = False
                        tts_queue.put(f"Rest complete. Starting set {current_set}")
                        vision_loop.last_rest_announcement = None

                # MediaPipe pose detection
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(rgb_frame)

                head_detected = False
                head_x = 0
                if pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks.landmark
                    nose = landmarks[mp_pose.PoseLandmark.NOSE]
                    head_x = int(nose.x * w)
                    head_y = int(nose.y * h)
                    head_detected = True

                    if not args.headless:
                        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        cv2.circle(frame, (head_x, head_y), 10, (255, 0, 255), -1)
                        cv2.putText(frame, "HEAD", (head_x+15, head_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                # YOLO detection
                try:
                    results = model(frame, conf=yolo_conf_threshold, verbose=False)
                except Exception as e:
                    log_event("ERROR", f"YOLO inference error: {e}")
                    break

                person_detected = False
                person_top_y = h
                
                # We need cx and cy of ball for logging later
                ball_cx = None
                ball_cy = None
                quadrant = None

                for r in results:
                    boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                    classes = r.boxes.cls.cpu().numpy().astype(int)

                    for box, cls in zip(boxes, classes):
                        x1, y1, x2, y2 = box
                        cx, cy = (x1+x2)//2, (y1+y2)//2
                        label = model.names[int(cls)] if int(cls) < len(model.names) else "unknown"

                        if label == "person":
                            person_x, person_y = cx, cy
                            person_top_y = y1
                            person_detected = True
                            if not args.headless:
                                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 3)
                                cv2.putText(frame, "PERSON", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

                        elif label == "ball":
                            ball_cx, ball_cy = cx, cy
                            if not args.headless:
                                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
                                cv2.putText(frame, "BALL", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                            ball_overhead = False
                            if head_detected:
                                ball_overhead = cy < head_y
                                if not args.headless:
                                    status_txt = "OVERHEAD ✓" if ball_overhead else "TOO LOW ✗"
                                    color = (0, 255, 0) if ball_overhead else (0, 0, 255)
                                    cv2.putText(frame, status_txt, (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            elif person_detected:
                                ball_overhead = cy < person_top_y
                                if not args.headless:
                                    status_txt = "OVERHEAD ✓" if ball_overhead else "TOO LOW ✗"
                                    color = (0, 255, 0) if ball_overhead else (0, 0, 255)
                                    cv2.putText(frame, status_txt, (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                            if person_detected:
                                if cy < person_y:  # Top half
                                    if ball_overhead:
                                        quadrant = TOP_RIGHT if cx > person_x else TOP_LEFT
                                    else:
                                        continue # ball not high enough
                                else:  # Bottom half
                                    quadrant = BOTTOM_LEFT if cx < person_x else BOTTOM_RIGHT

                                if not ball_sequence or quadrant != ball_sequence[-1]:
                                    ball_sequence.append(quadrant)
                                    log_event("QUADRANT", f"Ball moved to quadrant {quadrant}", {"sequence": ball_sequence})

                # Visuals for quadrants
                if not args.headless and person_detected:
                    cv2.line(frame, (0, person_y), (w, person_y), (0,0,255), 3)
                    cv2.line(frame, (person_x, 0), (person_x, h), (255,0,0), 3)
                    cv2.putText(frame, "OVERHEAD-NEAR", (person_x+20, person_y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.putText(frame, "OVERHEAD-FAR", (20, person_y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.putText(frame, "RELEASED", (20, person_y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                    cv2.putText(frame, "WRONG", (person_x+20, person_y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                # Game Logic Check
                if len(ball_sequence) >= 3:
                    if (ball_sequence[0] == TOP_RIGHT and 
                        ball_sequence[1] == TOP_LEFT and 
                        ball_sequence[2] == BOTTOM_LEFT):
                        
                        rep_count += 1
                        log_event("REP_COMPLETE", f"Rep {rep_count}/{reps_per_set}")
                        tts_queue.put(f"Rep {rep_count} complete!")

                        if rep_count >= reps_per_set:
                            current_set += 1
                            rep_count = 0

                            if current_set <= sets:
                                tts_queue.put(f"Set {current_set - 1} complete. Rest for {rest_between_sets} seconds.")
                                is_resting = True
                                rest_start_time = time.time()
                            else:
                                tts_queue.put("All sets complete! Great job!")
                                stop_event.set()
                                break
                        else:
                            tts_queue.put(f"Pause for {pause_between_reps} seconds")
                            is_paused = True
                            pause_start_time = time.time()

                        ball_sequence = []

                # Break Cases Check
                if len(ball_sequence) >= 2:
                    break_triggered = False
                    if ball_sequence[0] == TOP_RIGHT and ball_sequence[1] == BOTTOM_LEFT:
                        log_event("BREAK", "Direct throw without rotation")
                        tts_queue.put("Rotate the ball overhead to the left before throwing.")
                        break_triggered = True
                    elif ball_sequence[0] == TOP_LEFT and ball_sequence[1] == TOP_RIGHT:
                        log_event("BREAK", "Wrong rotation direction")
                        tts_queue.put("Rotate from right to left, not left to right.")
                        break_triggered = True
                    elif ball_sequence[0] == TOP_RIGHT and ball_sequence[1] == BOTTOM_RIGHT:
                        log_event("BREAK", "Incomplete rotation")
                        tts_queue.put("Complete the overhead rotation to the left before throwing.")
                        break_triggered = True
                    elif ball_sequence[0] == TOP_LEFT:
                        log_event("BREAK", "Wrong starting position")
                        tts_queue.put("Start with ball overhead on your right side.")
                        break_triggered = True
                    elif ball_sequence[0] == TOP_RIGHT and ball_sequence[1] == TOP_LEFT:
                        if len(ball_sequence) >= 3 and ball_sequence[2] in [TOP_LEFT, TOP_RIGHT]:
                            log_event("BREAK", "No throw detected")
                            tts_queue.put("Throw the ball after rotating.")
                            break_triggered = True

                    if break_triggered:
                        ball_sequence = []

                # Data buffer
                data_buffer.append({
                    "timestamp": datetime.now().isoformat(),
                    "frame": frame_counter,
                    "set": current_set,
                    "rep": rep_count,
                    "ball_x": ball_cx,
                    "ball_y": ball_cy,
                    "person_x": person_x if person_detected else None,
                    "person_y": person_y if person_detected else None,
                    "head_y": head_y if head_detected else None,
                    "current_quadrant": quadrant
                })

                # Visuals
                if not args.headless:
                    if person_detected:
                        cv2.putText(frame, f"Set: {current_set}/{sets}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                        cv2.putText(frame, f"Rep: {rep_count}/{reps_per_set}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                        cv2.putText(frame, f"Sequence: {ball_sequence}", (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                        cv2.putText(frame, "Camera: RIGHT SIDE VIEW", (w-350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    else:
                        cv2.putText(frame, "PERSON NOT DETECTED", (w//2-200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

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
        tts_queue.put("__STOP__")
        log_event("CLEANUP", "Camera released")
        save_data(data_buffer, args.output, args.game_id, "rotationtoss")

    log_event("COMPLETED", "Rotational Toss ended", {"total_sets": current_set})


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
    parser = argparse.ArgumentParser(description="Rotational Toss")
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
