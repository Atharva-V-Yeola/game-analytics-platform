#!/usr/bin/env python3
"""
Game: Obstacle Sprint Dash (Exercise 16)
Description: Relay sprint with cones (zones), hurdles (bounce timing), and crawling zones.
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
    """Dedicated TTS thread — non-blocking. COM-safe for Windows."""
    if not TTS_AVAILABLE:
        log_event("INFO", "TTS not available")
        return

    # Windows COM requires explicit per-thread initialization
    if platform.system() == "Windows":
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass

    tts_driver = config.get("tts_driver")
    rate = config.get("tts_rate", 160)
    volume = config.get("tts_volume", 1.0)

    while not stop_event.is_set():
        try:
            msg = tts_queue.get(timeout=0.5)
            if msg == "__STOP__":
                break
            # Re-init engine per utterance to prevent COM/SAPI5 deadlock on Windows
            engine = init_tts(tts_driver)
            if engine:
                engine.setProperty('rate', rate)
                engine.setProperty('volume', volume)
                engine.say(msg)
                engine.runAndWait()
                engine.stop()
                del engine
        except queue.Empty:
            continue
        except Exception as e:
            log_event("ERROR", f"TTS error: {e}")

    # Windows COM cleanup
    if platform.system() == "Windows":
        try:
            import pythoncom
            pythoncom.CoUninitialize()
        except ImportError:
            pass

    log_event("CLEANUP", "TTS thread ended")


# =============================================================================
# DETECTION HELPERS
# =============================================================================

def get_person_and_ball(results, model_obj, person_conf_thresh, ball_conf_thresh):
    best_person_conf = 0.0
    person_box = person_cx = person_cy = None

    best_ball_conf = 0.0
    ball_box = ball_cx = ball_cy = None

    for r in results:
        if r.boxes is None:
            continue
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        clss  = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for box, cls, conf in zip(boxes, clss, confs):
            name = model_obj.names[int(cls)]
            x1, y1, x2, y2 = box
            if name == "person" and conf > person_conf_thresh and conf > best_person_conf:
                best_person_conf = conf
                person_box = box
                person_cx  = (x1 + x2) // 2
                person_cy  = (y1 + y2) // 2
            elif name == "ball" and conf > ball_conf_thresh and conf > best_ball_conf:
                best_ball_conf = conf
                ball_box = box
                ball_cx  = (x1 + x2) // 2
                ball_cy  = (y1 + y2) // 2

    return person_box, person_cx, person_cy, ball_box, ball_cx, ball_cy


# =============================================================================
# CRAWL DETECTOR
# =============================================================================

class CrawlDetector:
    def __init__(self, crawl_line_y: int, fw: int, config: dict):
        self.crawl_line_y  = crawl_line_y
        self.fw            = fw
        self.zone_x1       = int(fw * config.get("crawl_zone_x_start", 0.33))
        self.zone_x2       = int(fw * config.get("crawl_zone_x_end", 0.67))
        self.min_dip_secs  = config.get("crawl_min_dip_secs", 0.4)

        self.dip_start     = None
        self.was_below     = False
        self.crawl_count   = 0
        self.last_crawl_t  = 0.0

    def update(self, pose_landmarks, fh: int, fw: int) -> bool:
        if pose_landmarks is None:
            return False

        if not MEDIAPIPE_AVAILABLE:
            return False

        lms  = pose_landmarks.landmark
        PL   = mp.solutions.pose.PoseLandmark
        nose = lms[PL.NOSE]

        if nose.visibility < 0.50:
            return False

        nx = int(nose.x * fw)
        ny = int(nose.y * fh)

        if not (self.zone_x1 <= nx <= self.zone_x2):
            self.dip_start  = None
            self.was_below  = False
            return False

        below = ny > self.crawl_line_y
        now   = time.time()

        if below and not self.was_below:
            self.dip_start  = now
            self.was_below  = True
        elif not below and self.was_below:
            if (self.dip_start is not None
                    and now - self.dip_start >= self.min_dip_secs
                    and now - self.last_crawl_t > 1.5):
                self.crawl_count  += 1
                self.last_crawl_t  = now
                self.dip_start     = None
                self.was_below     = False
                log_event("CRAWL", f"Crawl #{self.crawl_count} detected")
                return True
            self.was_below = False
            self.dip_start = None

        return False

    def draw(self, frame, pose_landmarks, fh: int, fw: int):
        ov = frame.copy()
        cv2.rectangle(ov, (self.zone_x1, 0), (self.zone_x2, fh), (180, 80, 255), -1)
        cv2.addWeighted(ov, 0.12, frame, 0.88, 0, frame)
        cv2.rectangle(frame, (self.zone_x1, 0), (self.zone_x2, fh), (180, 80, 255), 2)
        cv2.putText(frame, "CRAWL ZONE", (self.zone_x1 + 6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 80, 255), 2)

        cv2.line(frame, (self.zone_x1, self.crawl_line_y), (self.zone_x2, self.crawl_line_y), (255, 80, 255), 3)
        cv2.putText(frame, "Crawl line (duck below)", (self.zone_x1 + 6, self.crawl_line_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 80, 255), 2)

        if pose_landmarks and MEDIAPIPE_AVAILABLE:
            lms  = pose_landmarks.landmark
            PL   = mp.solutions.pose.PoseLandmark
            nose = lms[PL.NOSE]
            if nose.visibility >= 0.50:
                nx = int(nose.x * fw)
                ny = int(nose.y * fh)
                below = ny > self.crawl_line_y
                col   = (0, 0, 255) if below else (0, 255, 180)
                cv2.circle(frame, (nx, ny), 10, col, -1)
                label = "DUCKING!" if below else "face"
                cv2.putText(frame, label, (nx + 12, ny), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        cv2.putText(frame, f"Crawls: {self.crawl_count}", (self.zone_x1 + 6, self.crawl_line_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 80, 255), 2)


# =============================================================================
# BOUNCE ZONE DETECTOR
# =============================================================================

class BounceZoneDetector:
    def __init__(self, fw: int, fh: int, config: dict):
        self.vertical_line  = fw // 2
        self.bounce_line_y  = int(fh * config.get("bounce_line_ratio", 0.60))
        self.bounces_per_zone = config.get("bounces_per_zone", 5)
        self.bounce_debounce = config.get("bounce_debounce", 0.5)
        self.reset()

    def reset(self):
        self.left_count      = 0
        self.right_count     = 0
        self.last_zone       = None
        self.ball_was_above  = True
        self.last_bounce_t   = 0.0
        self.violation       = False
        self.violation_msg   = ""
        self.complete        = False
        self.timer_started   = False
        self.set_start_t     = 0.0

    def update(self, ball_cx, ball_cy) -> str:
        if ball_cx is None or ball_cy is None:
            return None

        now        = time.time()
        ball_zone  = "LEFT" if ball_cx < self.vertical_line else "RIGHT"
        below      = ball_cy > self.bounce_line_y

        if below and self.ball_was_above:
            if now - self.last_bounce_t < self.bounce_debounce:
                self.ball_was_above = not below
                return None

            if not self.timer_started:
                self.timer_started = True
                self.set_start_t   = now

            if ball_zone == self.last_zone:
                self.violation     = True
                self.violation_msg = f"Must alternate zones! ({ball_zone} again)"
                log_event("VIOLATION", self.violation_msg)
                return "violation"

            if ball_zone == "LEFT":
                self.left_count  += 1
                log_event("BOUNCE", f"LEFT #{self.left_count}")
            else:
                self.right_count += 1
                log_event("BOUNCE", f"RIGHT #{self.right_count}")

            self.last_zone    = ball_zone
            self.last_bounce_t = now

            if (self.left_count  > self.bounces_per_zone or self.right_count > self.bounces_per_zone):
                self.violation     = True
                self.violation_msg = f"Too many bounces in {ball_zone} zone!"
                log_event("VIOLATION", self.violation_msg)
                return "violation"

            if (self.left_count  == self.bounces_per_zone and self.right_count == self.bounces_per_zone):
                self.complete = True
                return "complete"

            self.ball_was_above = False
            return f"bounce_{ball_zone.lower()}"

        if not below:
            self.ball_was_above = True

        return None

    def elapsed(self) -> float:
        if not self.timer_started:
            return 0.0
        return time.time() - self.set_start_t

    def draw(self, frame, ball_cx, ball_cy, fw, fh):
        cv2.line(frame, (self.vertical_line, 0), (self.vertical_line, fh), (255, 255, 0), 2)
        cv2.line(frame, (0, self.bounce_line_y), (self.vertical_line, self.bounce_line_y), (0, 255, 0), 2)
        cv2.line(frame, (self.vertical_line, self.bounce_line_y), (fw, self.bounce_line_y), (0, 80, 255), 2)

        cv2.putText(frame, "LEFT ZONE",  (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "RIGHT ZONE", (self.vertical_line + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)

        if ball_cx and ball_cy:
            col = (0, 255, 0) if ball_cx < self.vertical_line else (0, 80, 255)
            cv2.circle(frame, (ball_cx, ball_cy), 12, col, -1)


# =============================================================================
# HUD HELPERS
# =============================================================================

def draw_info_panel(frame, cur_set, bounce: BounceZoneDetector, crawl: CrawlDetector, fh: int, config: dict):
    px, py = 10, fh - 220
    bounces_per_zone = config.get("bounces_per_zone", 5)
    sets = config.get("sets", 4)
    
    cv2.rectangle(frame, (px, py), (px + 380, fh - 8), (0, 0, 0), -1)
    cv2.rectangle(frame, (px, py), (px + 380, fh - 8), (80, 80, 80), 1)

    def txt(msg, dy, col=(255, 255, 255), scale=0.65, thick=1):
        cv2.putText(frame, msg, (px + 10, py + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick)

    txt(f"Set: {cur_set}/{sets}  |  Time: {bounce.elapsed():.1f}s", 28, (0, 255, 255), 0.7, 2)

    l_col = (0, 255, 0) if bounce.left_count  == bounces_per_zone else (200, 200, 200)
    r_col = (0, 255, 0) if bounce.right_count == bounces_per_zone else (200, 200, 200)
    txt(f"LEFT  bounces: {bounce.left_count}/{bounces_per_zone}",  62, l_col)
    txt(f"RIGHT bounces: {bounce.right_count}/{bounces_per_zone}", 90, r_col)

    last = bounce.last_zone or "-"
    txt(f"Last zone: {last}  |  Must alternate!", 118, (255, 255, 100))
    txt(f"Crawls detected: {crawl.crawl_count}", 146, (180, 80, 255))

    if bounce.violation:
        txt(bounce.violation_msg, 178, (0, 0, 255), 0.55, 2)

def draw_large_msg(frame, text: str, col, fw: int, fh: int):
    cv2.putText(frame, text, (fw // 2 - len(text) * 9, fh // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3)

def draw_setup_screen(frame, calib_progress: float, nose_y_avg, fw: int, fh: int, config: dict):
    sets = config.get("sets", 4)
    bounces_per_zone = config.get("bounces_per_zone", 5)
    crawl_factor = config.get("crawl_line_factor", 1.45)
    
    cv2.putText(frame, "OBSTACLE SPRINT DASH - SETUP", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    lines = [
        "1. Stand 6-8 ft from camera, full body visible",
        "2. Camera at 30-45 deg side angle (3/4 view)",
        "3. Centre third of frame = crawl zone (purple band)",
        "4. Bounce ball alternating LEFT and RIGHT zones",
        f"5. Target: {sets} sets x {bounces_per_zone} bounces/zone",
    ]
    for i, l in enumerate(lines):
        cv2.putText(frame, l, (20, 85 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 1)

    if calib_progress > 0:
        bar_w = int((fw - 40) * min(calib_progress, 1.0))
        cv2.rectangle(frame, (20, fh - 80), (fw - 20, fh - 55), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, fh - 80), (20 + bar_w, fh - 55), (0, 255, 180), -1)
        cv2.putText(frame, f"Calibrating standing height... {int(calib_progress*100)}%", (20, fh - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 1)
        if nose_y_avg:
            cv2.putText(frame, f"Nose baseline: {nose_y_avg:.0f}px  |  Crawl line: {nose_y_avg * crawl_factor:.0f}px", (20, fh - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1)
    else:
        cv2.putText(frame, "Press 's' - stand upright and face camera for calibration", (20, fh - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def draw_rest_screen(frame, remaining: float, next_set: int, set_times: list, fw: int, fh: int, config: dict):
    sets = config.get("sets", 4)
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (fw, fh), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "REST", (fw // 2 - 80, fh // 2 - 60), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 4)
    cv2.putText(frame, f"{int(remaining)}s", (fw // 2 - 50, fh // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
    cv2.putText(frame, f"Next: Set {next_set}/{sets}", (fw // 2 - 120, fh // 2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (180, 180, 180), 2)
    for i, t in enumerate(set_times):
        cv2.putText(frame, f"  Set {i+1}: {t:.1f}s", (fw // 2 - 60, fh // 2 + 120 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)


# =============================================================================
# MAIN VISION LOOP
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
    sets = config.get("sets", 4)
    rest_time = config.get("rest_time_seconds", 105)
    calibration_secs = config.get("calibration_secs", 3)
    crawl_line_factor = config.get("crawl_line_factor", 1.45)
    ball_conf = config.get("ball_conf", 0.55)
    person_conf = config.get("person_conf", 0.50)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    setup_mode    = not args.headless
    setup_start_time = time.time()
    calibrating   = False
    calib_start   = 0.0
    nose_samples  = []
    nose_baseline = None
    crawl_line_y  = None

    cur_set       = 1
    in_rest       = False
    rest_start    = 0.0
    session_done  = False
    set_times     = []

    bounce = None
    crawl  = None

    frame_counter = 0
    prev_time     = time.time()
    fps           = 0.0
    rest_announced = set()
    
    data_buffer = []

    log_event("READY", "Obstacle Sprint Dash active", {"sets": sets})
    
    if setup_mode:
        tts_queue.put("Obstacle Sprint Dash. Stand upright and press S to calibrate.")
    else:
        # Headless mock calibration
        ret, test_frame = cap.read()
        if ret:
            fh, fw = test_frame.shape[:2]
            nose_baseline = int(fh * 0.3)
            crawl_line_y = int(nose_baseline * crawl_line_factor)
            bounce = BounceZoneDetector(fw, fh, config)
            crawl  = CrawlDetector(crawl_line_y, fw, config)
            setup_mode = False
            tts_queue.put("Calibration complete. Crawl line set. Starting set 1. Go!")

    try:
        with mp_pose.Pose(min_detection_confidence=0.55, min_tracking_confidence=0.55) as pose:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    log_event("WARN", "No frame detected, ending loop")
                    break

                frame_counter += 1
                fh, fw = frame.shape[:2]

                if frame_counter % 10 == 0:
                    now = time.time()
                    fps = 10 / max(now - prev_time, 0.001)
                    prev_time = now

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(rgb)
                pose_landmarks = pose_results.pose_landmarks

                if setup_mode:
                    # Auto-trigger calibration after 5 seconds (for backend-launched processes)
                    if not calibrating and time.time() - setup_start_time >= 5:
                        calibrating  = True
                        calib_start  = time.time()
                        nose_samples = []
                        tts_queue.put("Auto-calibrating. Stand still, looking forward.")

                    if not args.headless:
                        calib_progress = 0.0
                        nose_y_avg = None

                        if calibrating:
                            elapsed_c = time.time() - calib_start
                            calib_progress = elapsed_c / calibration_secs

                            if pose_landmarks:
                                PL = mp_pose.PoseLandmark
                                nose = pose_landmarks.landmark[PL.NOSE]
                                if nose.visibility >= 0.55:
                                    nose_samples.append(nose.y * fh)

                            if elapsed_c >= calibration_secs:
                                if len(nose_samples) >= 5:
                                    nose_baseline = float(np.mean(nose_samples))
                                    crawl_line_y  = int(nose_baseline * crawl_line_factor)

                                    bounce = BounceZoneDetector(fw, fh, config)
                                    crawl  = CrawlDetector(crawl_line_y, fw, config)

                                    setup_mode  = False
                                    calibrating = False
                                    tts_queue.put("Calibration complete. Crawl line set. Starting set 1. Go!")
                                    log_event("CALIBRATION", "Complete", {"nose_baseline": nose_baseline, "crawl_line": crawl_line_y})
                                else:
                                    tts_queue.put("Could not detect face. Stand closer and try again.")
                                    calibrating  = False
                                    nose_samples = []
                        else:
                            nose_y_avg = float(np.mean(nose_samples)) if nose_samples else None

                        if pose_landmarks:
                            mp_drawing.draw_landmarks(
                                frame, pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(180,180,180), thickness=1, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(100,100,200), thickness=1),
                            )

                        draw_setup_screen(frame, calib_progress, nose_y_avg, fw, fh, config)
                        cv2.putText(frame, f"FPS:{int(fps)}", (fw - 80, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
                        cv2.imshow(args.game_id, frame)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('s') and not calibrating:
                            calibrating  = True
                            calib_start  = time.time()
                            nose_samples = []
                            tts_queue.put("Calibrating. Stand still, looking forward.")
                        elif key == ord('q'):
                            stop_event.set()
                            break
                    else:
                        # Headless calibration path (auto-triggered)
                        if calibrating:
                            elapsed_c = time.time() - calib_start
                            if pose_landmarks:
                                PL = mp_pose.PoseLandmark
                                nose = pose_landmarks.landmark[PL.NOSE]
                                if nose.visibility >= 0.55:
                                    nose_samples.append(nose.y * fh)
                            if elapsed_c >= calibration_secs:
                                if len(nose_samples) >= 5:
                                    nose_baseline = float(np.mean(nose_samples))
                                    crawl_line_y  = int(nose_baseline * crawl_line_factor)
                                    bounce = BounceZoneDetector(fw, fh, config)
                                    crawl  = CrawlDetector(crawl_line_y, fw, config)
                                    setup_mode  = False
                                    calibrating = False
                                    tts_queue.put("Calibration complete. Crawl line set. Starting set 1. Go!")
                                    log_event("CALIBRATION", "Complete", {"nose_baseline": nose_baseline, "crawl_line": crawl_line_y})
                                else:
                                    # Fallback: use default nose baseline
                                    nose_baseline = int(fh * 0.3)
                                    crawl_line_y  = int(nose_baseline * crawl_line_factor)
                                    bounce = BounceZoneDetector(fw, fh, config)
                                    crawl  = CrawlDetector(crawl_line_y, fw, config)
                                    setup_mode  = False
                                    calibrating = False
                                    tts_queue.put("Using default calibration. Starting set 1. Go!")
                                    log_event("CALIBRATION", "Default fallback", {"nose_baseline": nose_baseline})
                    continue

                if in_rest:
                    elapsed_r  = time.time() - rest_start
                    remaining  = max(0.0, rest_time - elapsed_r)

                    for m in (90, 60, 45, 30, 15, 10, 5):
                        if remaining <= m and m not in rest_announced:
                            rest_announced.add(m)
                            tts_queue.put(f"{m} seconds.")

                    if not args.headless:
                        draw_rest_screen(frame, remaining, cur_set, set_times, fw, fh, config)
                        cv2.putText(frame, f"FPS:{int(fps)}", (fw - 80, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
                        cv2.imshow(args.game_id, frame)
                    else:
                        time.sleep(0.01)

                    if elapsed_r >= rest_time:
                        in_rest        = False
                        rest_announced = set()
                        bounce.reset()
                        tts_queue.put(f"Rest over. Set {cur_set}. Go!")
                        log_event("REST_END", f"Starting set {cur_set}")

                    if not args.headless and cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_event.set()
                        break
                    continue

                if session_done:
                    if not args.headless:
                        draw_large_msg(frame, "SESSION COMPLETE!", (0, 255, 0), fw, fh)
                        best_idx = set_times.index(min(set_times)) + 1 if set_times else None
                        for i, t in enumerate(set_times):
                            mark = " <<BEST" if (i + 1) == best_idx else ""
                            cv2.putText(frame, f"Set {i+1}: {t:.1f}s{mark}", (fw // 2 - 80, fh // 2 + 40 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180) if mark else (200, 200, 200), 1)
                        cv2.imshow(args.game_id, frame)
                        cv2.waitKey(4000)
                    stop_event.set()
                    break

                try:
                    yolo_results = model(frame, verbose=False, conf=ball_conf)
                except Exception as e:
                    log_event("ERROR", f"YOLO error: {e}")
                    break
                    
                _, _, _, ball_box, ball_cx, ball_cy = get_person_and_ball(yolo_results, model, person_conf, ball_conf)

                if pose_landmarks and not args.headless:
                    mp_drawing.draw_landmarks(
                        frame, pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(160,160,160), thickness=1, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(80, 80, 180), thickness=1),
                    )

                event = bounce.update(ball_cx, ball_cy)

                if event == "violation":
                    tts_queue.put(f"Violation! {bounce.violation_msg}")
                    if not args.headless:
                        draw_large_msg(frame, "VIOLATION!", (0, 0, 255), fw, fh)
                        bounce.draw(frame, ball_cx, ball_cy, fw, fh)
                        crawl.draw(frame, pose_landmarks, fh, fw)
                        draw_info_panel(frame, cur_set, bounce, crawl, fh, config)
                        cv2.imshow(args.game_id, frame)
                        cv2.waitKey(2000)
                    tts_queue.put(f"Restarting set {cur_set}.")
                    bounce.reset()

                elif event == "complete":
                    set_time = bounce.elapsed()
                    set_times.append(set_time)
                    tts_queue.put(f"Set {cur_set} complete! Time: {set_time:.1f} seconds.")
                    log_event("SET_COMPLETE", f"Set {cur_set} time: {set_time:.1f}s")
                    
                    if not args.headless:
                        draw_large_msg(frame, f"SET {cur_set} DONE  {set_time:.1f}s", (0, 255, 0), fw, fh)
                        cv2.imshow(args.game_id, frame)
                        cv2.waitKey(2000)

                    if cur_set < sets:
                        cur_set   += 1
                        in_rest    = True
                        rest_start = time.time()
                        tts_queue.put(f"Rest for {rest_time} seconds.")
                    else:
                        best_t   = min(set_times)
                        best_idx = set_times.index(best_t) + 1
                        tts_queue.put(f"All {sets} sets complete! Best set was set {best_idx} at {best_t:.1f} seconds. Outstanding work!")
                        log_event("COMPLETED", "All sets complete")
                        session_done = True

                elif event and event.startswith("bounce"):
                    zone = "LEFT" if "left" in event else "RIGHT"
                    tts_queue.put(f"{zone} {bounce.left_count if zone == 'LEFT' else bounce.right_count}")

                crawl_done = crawl.update(pose_landmarks, fh, fw)
                if crawl_done:
                    tts_queue.put("Crawl!")
                    
                data_buffer.append({
                    "timestamp": datetime.now().isoformat(),
                    "frame": frame_counter,
                    "set": cur_set,
                    "elapsed_time": bounce.elapsed(),
                    "left_count": bounce.left_count,
                    "right_count": bounce.right_count,
                    "crawl_count": crawl.crawl_count
                })

                if not args.headless:
                    bounce.draw(frame, ball_cx, ball_cy, fw, fh)
                    crawl.draw(frame, pose_landmarks, fh, fw)
                    draw_info_panel(frame, cur_set, bounce, crawl, fh, config)

                    if nose_baseline:
                        cv2.line(frame, (0, int(nose_baseline)), (fw, int(nose_baseline)), (100, 100, 100), 1)
                        cv2.putText(frame, "standing head level", (4, int(nose_baseline) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

                    person_results = yolo_results
                    for r in person_results:
                        if r.boxes is None:
                            continue
                        for box, cls, conf in zip(r.boxes.xyxy.cpu().numpy().astype(int), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
                            if model.names[int(cls)] == "person" and conf > person_conf:
                                x1, y1, x2, y2 = box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 0), 2)

                    cv2.putText(frame, f"Set {cur_set}/{sets}  |  L:{bounce.left_count}  R:{bounce.right_count}  |  {bounce.elapsed():.1f}s", (fw - 420, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
                    cv2.putText(frame, f"FPS:{int(fps)}", (fw - 80, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

                    cv2.imshow(args.game_id, frame)

                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('q'):
                        stop_event.set()
                        break
                    elif key == ord('r'):
                        tts_queue.put("Resetting session.")
                        cur_set      = 1
                        set_times    = []
                        in_rest      = False
                        session_done = False
                        rest_announced = set()
                        bounce.reset()
                        crawl.crawl_count = 0
                else:
                    time.sleep(0.01)

    except Exception as e:
        log_event("ERROR", f"Vision loop exception: {str(e)}")
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()
        save_data(data_buffer, args.output, args.game_id, "sprintdash")
        log_event("CLEANUP", "Camera released", {"set_times": set_times})


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
    parser = argparse.ArgumentParser(description="Obstacle Sprint Dash")
    parser.add_argument("--game-id", required=True, help="Unique game identifier")
    parser.add_argument("--camera", default="0", help="Camera index or video file")
    parser.add_argument("--output", default="./data", help="CSV output directory")
    parser.add_argument("--config", default=None, help="Path to JSON config")
    parser.add_argument("--model-path", default="models/all_in_one_yolov82/weights/best.pt", help="YOLO model")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")

    args = parser.parse_args()
    config = load_config(args.config)

    log_event("INIT", "Game starting", {"game_id": args.game_id})

    t_tts = threading.Thread(target=tts_worker, args=(config,))
    t_tts.start()

    vision_loop(args, config)

    tts_queue.put("__STOP__")
    t_tts.join(timeout=5.0)

    log_event("COMPLETED", "Program stopped gracefully")


if __name__ == "__main__":
    main()


