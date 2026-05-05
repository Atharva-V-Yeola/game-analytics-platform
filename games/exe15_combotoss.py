#!/usr/bin/env python3
"""
Game: Combo Power Toss (Exercise 15)
Description: Sequence — chest throw -> overhead rainbow -> rotational toss.
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
# BALL UTILS
# =============================================================================

def is_valid_ball(box, fh, fw):
    x1, y1, x2, y2 = box
    w = x2 - x1; h = y2 - y1
    area = w * h
    fa   = fw * fh
    if area < fa * 0.001 or area > fa * 0.15:
        return False
    cx = (x1 + x2) // 2
    if cx < fw * 0.05 or cx > fw * 0.95:
        return False
    ar = max(w, h) / max(min(w, h), 1)
    if ar > 2.2:
        return False
    return True


# =============================================================================
# PHASE 1 — CHEST THROW DETECTOR
# =============================================================================

class ChestThrowDetector:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.seen_ids      = set()
        self.completed     = False
        self.last_cross_t  = 0.0

    def update(self, model, frame, fw, fh, yolo_results):
        if self.completed:
            return False

        line_ratio = self.config.get("chest_line_ratio", 0.45)
        min_gap = self.config.get("min_chest_cross_gap", 0.8)
        line_x = int(fw * line_ratio)
        
        # We need byte tracking for this phase specifically
        try:
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        except Exception as e:
            log_event("ERROR", f"YOLO tracking error: {e}")
            return False

        crossed = False
        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            clss  = r.boxes.cls.cpu().numpy().astype(int)
            ids   = r.boxes.id.cpu().numpy().astype(int)
            for box, cls, oid in zip(boxes, clss, ids):
                if model.names[int(cls)] != "ball":
                    continue
                if not is_valid_ball(box, fh, fw):
                    continue
                cx = (box[0] + box[2]) // 2
                cy = (box[1] + box[3]) // 2
                now = time.time()
                if (cx > line_x
                        and oid not in self.seen_ids
                        and now - self.last_cross_t > min_gap):
                    self.seen_ids.add(oid)
                    self.last_cross_t = now
                    self.completed    = True
                    crossed           = True
                    log_event("PHASE_1", "Chest throw detected")
                
                # We can draw it even if headless, just doesn't display
                cv2.circle(frame, (cx, cy), 12, (0, 255, 255), 2)

        cv2.line(frame, (line_x, 0), (line_x, fh), (255, 200, 0), 2)
        cv2.putText(frame, "Throw past line ->", (line_x - 200, fh // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
        return crossed

    def draw_hud(self, frame, fw, fh):
        status = "DONE" if self.completed else "Chest throw: pass ball across line"
        col    = (0, 255, 0) if self.completed else (0, 200, 255)
        cv2.putText(frame, status, (20, fh - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)


# =============================================================================
# QUADRANT HELPERS
# =============================================================================

Q_TOP_RIGHT    = 0
Q_TOP_LEFT     = 1
Q_BOTTOM_LEFT  = 2
Q_BOTTOM_RIGHT = 3
Q_NAMES        = ["TOP-RIGHT", "TOP-LEFT", "BOTTOM-LEFT", "BOTTOM-RIGHT"]

def get_quadrant(bx, by, px, py):
    if by < py:
        return Q_TOP_RIGHT    if bx > px else Q_TOP_LEFT
    else:
        return Q_BOTTOM_LEFT  if bx < px else Q_BOTTOM_RIGHT

def get_person_center(results, model, fw, fh, conf_thresh):
    best_conf = 0.0
    best_pt   = None
    for r in results:
        if r.boxes is None:
            continue
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        clss  = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for box, cls, conf in zip(boxes, clss, confs):
            if model.names[int(cls)] == "person" and conf > conf_thresh:
                if conf > best_conf:
                    best_conf = conf
                    best_pt   = ((box[0]+box[2])//2, (box[1]+box[3])//2)
    return best_pt

def get_ball_center(results, model, fh, fw, conf_thresh):
    best_conf = 0.0
    best_pt   = None
    for r in results:
        if r.boxes is None:
            continue
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        clss  = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for box, cls, conf in zip(boxes, clss, confs):
            if model.names[int(cls)] == "ball" and conf > conf_thresh:
                if is_valid_ball(box, fh, fw) and conf > best_conf:
                    best_conf = conf
                    best_pt   = ((box[0]+box[2])//2, (box[1]+box[3])//2)
    return best_pt

def draw_quadrant_overlay(frame, px, py, fw, fh, seq):
    cv2.line(frame, (0, py), (fw, py), (180, 180, 180), 1)
    cv2.line(frame, (px, 0), (px, fh), (180, 180, 180), 1)

    def highlight(x1, y1, x2, y2, col):
        ov = frame.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), col, -1)
        cv2.addWeighted(ov, 0.22, frame, 0.78, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

    if len(seq) == 0:
        highlight(px, 0, fw, py, (0, 255, 0))
        cv2.putText(frame, "START: TOP-RIGHT", (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    elif len(seq) == 1 and seq[0] == Q_TOP_RIGHT:
        highlight(0, 0, px, py, (0, 200, 255))
        cv2.putText(frame, "NEXT: TOP-LEFT", (10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    elif len(seq) >= 2 and seq[:2] == [Q_TOP_RIGHT, Q_TOP_LEFT]:
        highlight(0, py, px, fh, (0, 255, 200))
        cv2.putText(frame, "THROW: BOTTOM-LEFT", (10, py + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

    cv2.putText(frame, "TOP-R",  (px + 6,  20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, "TOP-L",  (6,        20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, "BOT-L",  (6,        py + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, "BOT-R",  (px + 6,  py + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


# =============================================================================
# PHASE 2 — OVERHEAD RAINBOW DETECTOR
# =============================================================================

class RainbowTossDetector:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.seq           = []
        self.completed     = False
        self.last_quad_t   = 0.0
        self.person_center = None
        self.ball_center   = None

    def update(self, results, model, fw, fh):
        if self.completed:
            return False

        p_conf = self.config.get("person_conf", 0.5)
        b_conf = self.config.get("ball_conf", 0.65)
        min_gap = self.config.get("min_quad_gap", 0.4)

        self.person_center = get_person_center(results, model, fw, fh, p_conf)
        self.ball_center   = get_ball_center(results, model, fh, fw, b_conf)

        if not self.person_center or not self.ball_center:
            return False

        px, py = self.person_center
        bx, by = self.ball_center
        quad   = get_quadrant(bx, by, px, py)
        now    = time.time()

        if not self.seq or (quad != self.seq[-1] and now - self.last_quad_t > min_gap):
            self.seq.append(quad)
            self.last_quad_t = now
            log_event("PHASE_2", f"Rainbow seq: {[Q_NAMES[q] for q in self.seq]}")

        if len(self.seq) >= 2:
            if self.seq[0] == Q_TOP_RIGHT and self.seq[1] == Q_BOTTOM_LEFT:
                tts_queue.put("No rotation — swing overhead first.")
                self.seq = []
                return False
            if self.seq[0] == Q_TOP_LEFT:
                tts_queue.put("Start from the right side.")
                self.seq = []
                return False
            if (len(self.seq) >= 2 and self.seq[0] == Q_TOP_RIGHT and self.seq[-1] == Q_BOTTOM_RIGHT):
                tts_queue.put("Wrong side — throw to the left.")
                self.seq = []
                return False

        if (len(self.seq) >= 3 and self.seq[0] == Q_TOP_RIGHT and self.seq[1] == Q_TOP_LEFT and self.seq[2] == Q_BOTTOM_LEFT):
            self.completed = True
            log_event("PHASE_2", "Rainbow toss complete")
            return True

        return False

    def draw_hud(self, frame, fw, fh):
        if self.person_center:
            draw_quadrant_overlay(frame, self.person_center[0], self.person_center[1], fw, fh, self.seq)
        if self.ball_center:
            cv2.circle(frame, self.ball_center, 12, (0, 255, 200), -1)

        seq_str = " -> ".join(Q_NAMES[q] for q in self.seq) if self.seq else "waiting..."
        status  = "DONE" if self.completed else f"Rainbow: {seq_str}"
        col     = (0, 255, 0) if self.completed else (0, 200, 255)
        cv2.putText(frame, status, (20, fh - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)


# =============================================================================
# PHASE 3 — ROTATIONAL TOSS DETECTOR
# =============================================================================

class RotationalTossDetector:
    def __init__(self, config, frame_ref):
        self.config = config
        self.frame_ref = frame_ref
        self.reset()

    def reset(self):
        self.seq            = []
        self.completed      = False
        self.last_quad_t    = 0.0
        self.person_center  = None
        self.ball_center    = None
        self.head_y         = None

    def update(self, results, model, pose_landmarks, fw, fh):
        if self.completed:
            return False

        p_conf = self.config.get("person_conf", 0.5)
        b_conf = self.config.get("ball_conf", 0.65)
        min_gap = self.config.get("min_quad_gap", 0.4)
        vis_thresh = self.config.get("visibility_thresh", 0.6)

        self.person_center = get_person_center(results, model, fw, fh, p_conf)
        self.ball_center   = get_ball_center(results, model, fh, fw, b_conf)

        if pose_landmarks and MEDIAPIPE_AVAILABLE:
            lms = pose_landmarks.landmark
            PL  = mp.solutions.pose.PoseLandmark
            nose = lms[PL.NOSE]
            if nose.visibility >= vis_thresh:
                self.head_y = int(nose.y * fh)

        if not self.person_center or not self.ball_center:
            return False

        px, py = self.person_center
        bx, by = self.ball_center

        if self.head_y is not None:
            if by >= self.head_y and by < py:
                if self.frame_ref[0] is not None:
                    cv2.putText(self.frame_ref[0], "Too low - raise ball overhead", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                return False

        quad = get_quadrant(bx, by, px, py)
        now  = time.time()

        if not self.seq or (quad != self.seq[-1] and now - self.last_quad_t > min_gap):
            self.seq.append(quad)
            self.last_quad_t = now
            log_event("PHASE_3", f"Rotational seq: {[Q_NAMES[q] for q in self.seq]}")

        if len(self.seq) >= 2:
            if self.seq[0] == Q_TOP_RIGHT and self.seq[1] == Q_BOTTOM_LEFT:
                tts_queue.put("Rotate overhead before throwing.")
                self.seq = []
                return False
            if self.seq[0] == Q_TOP_LEFT:
                tts_queue.put("Start with ball on your right side.")
                self.seq = []
                return False
            if (self.seq[0] == Q_TOP_RIGHT and self.seq[-1] == Q_BOTTOM_RIGHT):
                tts_queue.put("Throw forward to your left.")
                self.seq = []
                return False

        if (len(self.seq) >= 3 and self.seq[0] == Q_TOP_RIGHT and self.seq[1] == Q_TOP_LEFT and self.seq[2] == Q_BOTTOM_LEFT):
            self.completed = True
            log_event("PHASE_3", "Rotational toss complete")
            return True

        return False

    def draw_hud(self, frame, fw, fh):
        if self.person_center:
            draw_quadrant_overlay(frame, self.person_center[0], self.person_center[1], fw, fh, self.seq)
        if self.ball_center:
            cv2.circle(frame, self.ball_center, 12, (255, 100, 0), -1)
        if self.head_y is not None:
            cv2.line(frame, (0, self.head_y), (fw, self.head_y), (200, 100, 255), 1)
            cv2.putText(frame, "head level", (4, self.head_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 100, 255), 1)

        seq_str = " -> ".join(Q_NAMES[q] for q in self.seq) if self.seq else "waiting..."
        status  = "DONE" if self.completed else f"Rotational: {seq_str}"
        col     = (0, 255, 0) if self.completed else (255, 150, 50)
        cv2.putText(frame, status, (20, fh - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)


# =============================================================================
# MAIN GAME LOGIC
# =============================================================================

PHASE_NAMES   = ["Chest throw", "Rainbow toss", "Rotational toss"]
PHASE_COLORS  = [(255, 200, 0), (0, 200, 255), (255, 120, 50)]

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
    sets_total = config.get("sets", 3)
    reps_per_set = config.get("reps_per_set", 12)
    rest_time = config.get("rest_time_seconds", 75)
    frame_skip = config.get("frame_skip", 2)
    ball_conf = config.get("ball_conf", 0.65)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    setup_mode = not args.headless
    cur_set = 1
    cur_rep = 0
    in_rest = False
    rest_start = 0.0
    session_done = False

    frame_counter = 0
    frame_ref = [None]

    chest = ChestThrowDetector(config)
    rainbow = RainbowTossDetector(config)
    rotational = RotationalTossDetector(config, frame_ref)
    active_phase = 0

    data_buffer = []

    log_event("READY", "Combo Power Toss active", {
        "sets": sets_total,
        "reps_per_set": reps_per_set
    })

    if setup_mode:
        tts_queue.put("Combo Power Toss. Complete the setup then press S.")

    try:
        with mp_pose.Pose(min_detection_confidence=0.55, min_tracking_confidence=0.55) as pose:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    log_event("WARN", "No frame detected, ending loop")
                    break

                frame_counter += 1
                fh, fw = frame.shape[:2]
                frame_ref[0] = frame

                if setup_mode:
                    if not args.headless:
                        cv2.putText(frame, "COMBO POWER TOSS - SETUP", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                        lines = [
                            "1. Stand 6-8 feet from camera",
                            "2. Full body visible, floor visible",
                            "3. Camera at 30-45 degree side angle (3/4 view)",
                            "4. Sequence per rep: Chest pass -> Overhead rainbow -> Rotational toss",
                            f"5. Target: {reps_per_set} reps x {sets_total} sets",
                        ]
                        for i, l in enumerate(lines):
                            cv2.putText(frame, l, (20, 80 + i * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
                        cv2.putText(frame, "Press 's' to start", (20, fh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        cv2.imshow(args.game_id, frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('s'):
                            setup_mode = False
                            tts_queue.put("Starting set 1. First: chest throw, then rainbow, then rotational. Go!")
                        elif key == ord('q'):
                            stop_event.set()
                            break
                    continue

                if in_rest:
                    elapsed = time.time() - rest_start
                    remaining = max(0, rest_time - elapsed)

                    if not args.headless:
                        ov = frame.copy()
                        cv2.rectangle(ov, (0, 0), (fw, fh), (0, 0, 0), -1)
                        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
                        cv2.putText(frame, "REST", (fw // 2 - 70, fh // 2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 4)
                        cv2.putText(frame, f"{int(remaining)}s", (fw // 2 - 40, fh // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
                        cv2.putText(frame, f"Next: Set {cur_set}/{sets_total}", (fw // 2 - 110, fh // 2 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                        cv2.imshow(args.game_id, frame)
                    else:
                        time.sleep(0.01)

                    if int(remaining) in (60, 45, 30, 15, 10, 5) and getattr(vision_loop, "last_rest_announcement", None) != int(remaining):
                        tts_queue.put(f"{int(remaining)} seconds.")
                        vision_loop.last_rest_announcement = int(remaining)

                    if elapsed >= rest_time:
                        in_rest = False
                        active_phase = 0
                        chest.reset(); rainbow.reset(); rotational.reset()
                        tts_queue.put(f"Rest over. Set {cur_set}. Chest throw first!")
                        log_event("REST_END", f"Starting set {cur_set}")

                    if not args.headless and cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_event.set()
                        break
                    continue

                if session_done:
                    if not args.headless:
                        cv2.putText(frame, "SESSION COMPLETE!", (fw // 2 - 160, fh // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        cv2.imshow(args.game_id, frame)
                        cv2.waitKey(3000)
                    stop_event.set()
                    break

                yolo_results = None
                if frame_counter % frame_skip == 0 and active_phase > 0:
                    try:
                        yolo_results = model(frame, verbose=False, conf=ball_conf)
                    except Exception as e:
                        log_event("ERROR", f"YOLO error: {e}")
                        break

                pose_landmarks = None
                if active_phase == 2:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(rgb)
                    pose_landmarks = pose_results.pose_landmarks
                    if pose_landmarks and not args.headless:
                        mp_drawing.draw_landmarks(
                            frame, pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(180,180,180), thickness=1, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(100,100,200), thickness=1),
                        )

                if active_phase == 0:
                    done = chest.update(model, frame, fw, fh, yolo_results) # Needs full model for bytetrack
                    if done:
                        tts_queue.put("Chest throw done! Now overhead rainbow.")
                        active_phase = 1

                elif active_phase == 1:
                    if yolo_results is not None:
                        done = rainbow.update(yolo_results, model, fw, fh)
                        if done:
                            tts_queue.put("Rainbow done! Now rotational toss.")
                            active_phase = 2
                    if not args.headless:
                        rainbow.draw_hud(frame, fw, fh)

                elif active_phase == 2:
                    if yolo_results is not None:
                        done = rotational.update(yolo_results, model, pose_landmarks, fw, fh)
                        if done:
                            cur_rep += 1
                            tts_queue.put(f"Rep {cur_rep} complete!")
                            log_event("REP_COMPLETE", f"Rep {cur_rep} completed in set {cur_set}")

                            if cur_rep >= reps_per_set:
                                if cur_set < sets_total:
                                    tts_queue.put(f"Set {cur_set} done! Rest for {rest_time} seconds.")
                                    log_event("SET_COMPLETE", f"Set {cur_set} completed")
                                    cur_set += 1
                                    cur_rep = 0
                                    in_rest = True
                                    rest_start = time.time()
                                else:
                                    tts_queue.put("All sets complete! Outstanding combo work!")
                                    log_event("COMPLETED", "All sets complete")
                                    session_done = True
                            else:
                                active_phase = 0
                                chest.reset(); rainbow.reset(); rotational.reset()
                                tts_queue.put("Next rep. Chest throw!")

                    if not args.headless:
                        rotational.draw_hud(frame, fw, fh)

                data_buffer.append({
                    "timestamp": datetime.now().isoformat(),
                    "frame": frame_counter,
                    "set": cur_set,
                    "rep": cur_rep,
                    "active_phase": active_phase,
                    "chest_done": chest.completed,
                    "rainbow_done": rainbow.completed,
                    "rotational_done": rotational.completed
                })

                if not args.headless:
                    chest.draw_hud(frame, fw, fh)
                    
                    cv2.rectangle(frame, (0, 0), (fw, 120), (0, 0, 0), -1)
                    cv2.rectangle(frame, (0, 0), (fw, 120), (50, 50, 50), 1)
                    cv2.putText(frame, f"COMBO POWER TOSS  |  Set {cur_set}/{sets_total}  |  Rep {cur_rep}/{reps_per_set}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

                    for i, (name, col) in enumerate(zip(PHASE_NAMES, PHASE_COLORS)):
                        detectors = [chest, rainbow, rotational]
                        done = detectors[i].completed
                        active = (active_phase == i)
                        indicator = "[DONE]" if done else ("[<<]" if active else "[   ]")
                        text_col = (0, 255, 0) if done else (col if active else (120, 120, 120))
                        cv2.putText(frame, f"{indicator} {i+1}. {name}", (10, 58 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_col, 2 if active else 1)

                    cv2.imshow(args.game_id, frame)
                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('q'):
                        stop_event.set()
                        break
                    elif key == ord('r'):
                        tts_queue.put("Resetting session.")
                        cur_set = 1; cur_rep = 0
                        active_phase = 0
                        in_rest = False; session_done = False
                        chest.reset(); rainbow.reset(); rotational.reset()
                        setup_mode = True

    except Exception as e:
        log_event("ERROR", f"Vision loop exception: {str(e)}")
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()
        save_data(data_buffer, args.output, args.game_id, "combotoss")
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
    parser = argparse.ArgumentParser(description="Combo Power Toss")
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
