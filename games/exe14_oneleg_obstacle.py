#!/usr/bin/env python3
"""
Game: One-Leg Obstacle Bonus (Exercise 14)
Description: MediaPipe pose validation + YOLO ball tracking + wrist cone tapping.
"""

import argparse
import collections
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
# RULES AND MATH HELPERS
# =============================================================================

POSE_RULES = {
    "standing_leg": {
        "hip_angle":  (75,  105),
        "knee_angle": (160, 180),
    },
    "torso": {
        "angle":      (165, 180),
    },
    "free_leg": {
        "hip_angle":  (150, 200),
        "knee_angle": (160, 180),
    },
}

CONE_DEFS = [
    (0.15, 0.75, 0.10, 0.12),
    (0.85, 0.75, 0.10, 0.12),
    (0.50, 0.60, 0.10, 0.12),
]
CONE_COLORS_BGR = [
    (0,   140, 255),
    (255,  50,  50),
    (50,  200,  50),
]
CONE_TAP_HOLD_FRAMES = 3


def angle_between(a, b, c) -> float:
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))

def lm_xy(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)

def lm_vis(landmarks, idx):
    return landmarks[idx].visibility


class AngleSmoother:
    def __init__(self, window=5):
        self._bufs = collections.defaultdict(lambda: collections.deque(maxlen=window))

    def update(self, key: str, value: float) -> float:
        self._bufs[key].append(value)
        return float(np.mean(self._bufs[key]))


def check_visibility(landmarks, indices, threshold) -> bool:
    return all(lm_vis(landmarks, i) >= threshold for i in indices)

def validate_pose(landmarks, w, h, standing_side: str, smoother, config):
    if not MEDIAPIPE_AVAILABLE:
        return True, []
        
    tolerance = config.get("tolerance", 5)
    vis_thresh = config.get("visibility_threshold", 0.6)
    PL = mp.solutions.pose.PoseLandmark
    violations = []

    if standing_side == "left":
        s_shoulder = PL.LEFT_SHOULDER
        s_hip      = PL.LEFT_HIP
        s_knee     = PL.LEFT_KNEE
        s_ankle    = PL.LEFT_ANKLE
        f_hip      = PL.RIGHT_HIP
        f_knee     = PL.RIGHT_KNEE
        f_ankle    = PL.RIGHT_ANKLE
    else:
        s_shoulder = PL.RIGHT_SHOULDER
        s_hip      = PL.RIGHT_HIP
        s_knee     = PL.RIGHT_KNEE
        s_ankle    = PL.RIGHT_ANKLE
        f_hip      = PL.LEFT_HIP
        f_knee     = PL.LEFT_KNEE
        f_ankle    = PL.LEFT_ANKLE

    # Standing leg
    if check_visibility(landmarks, [s_shoulder, s_hip, s_knee, s_ankle], vis_thresh):
        sh = lm_xy(landmarks, s_shoulder, w, h)
        hi = lm_xy(landmarks, s_hip,      w, h)
        kn = lm_xy(landmarks, s_knee,     w, h)
        an = lm_xy(landmarks, s_ankle,    w, h)

        hip_ang  = smoother.update(f"sl_hip_{standing_side}",  angle_between(sh, hi, kn))
        knee_ang = smoother.update(f"sl_knee_{standing_side}", angle_between(hi, kn, an))

        lo, hi_ = POSE_RULES["standing_leg"]["hip_angle"]
        if not (lo - tolerance <= hip_ang <= hi_ + tolerance):
            violations.append(f"Standing hip angle {hip_ang:.0f} (want {lo}-{hi_})")

        lo, hi_ = POSE_RULES["standing_leg"]["knee_angle"]
        if not (lo - tolerance <= knee_ang <= hi_ + tolerance):
            violations.append(f"Standing knee bent ({knee_ang:.0f})")

    # Torso
    if check_visibility(landmarks, [s_shoulder, s_hip], vis_thresh):
        sh = lm_xy(landmarks, s_shoulder, w, h)
        hi = lm_xy(landmarks, s_hip,      w, h)
        vertical_ref = (hi[0], hi[1] - 100)
        torso_ang = smoother.update("torso", angle_between(sh, hi, vertical_ref))
        lo, hi_ = POSE_RULES["torso"]["angle"]
        if not (lo - tolerance <= torso_ang <= hi_ + tolerance):
            violations.append(f"Torso lean ({torso_ang:.0f})")

    # Free leg
    if check_visibility(landmarks, [f_hip, f_knee, f_ankle], vis_thresh):
        fhi = lm_xy(landmarks, f_hip,   w, h)
        fkn = lm_xy(landmarks, f_knee,  w, h)
        fan = lm_xy(landmarks, f_ankle, w, h)
        fhip_ang  = smoother.update(f"fl_hip_{standing_side}",  angle_between(fhi, fkn, fan))
        fknee_ang = smoother.update(f"fl_knee_{standing_side}", angle_between(fhi, fkn, fan))

        lo, hi_ = POSE_RULES["free_leg"]["hip_angle"]
        if not (lo - tolerance <= fhip_ang <= hi_ + tolerance):
            violations.append(f"Lift free leg higher (hip {fhip_ang:.0f})")

        lo, hi_ = POSE_RULES["free_leg"]["knee_angle"]
        if not (lo - tolerance <= fknee_ang <= hi_ + tolerance):
            violations.append(f"Free knee angle {fknee_ang:.0f} (want {lo}-{hi_})")

    return len(violations) == 0, violations

def is_valid_ball(box, fh, fw):
    x1, y1, x2, y2 = box
    w = x2 - x1; h = y2 - y1
    area = w * h
    fa = fw * fh
    if area < fa * 0.001 or area > fa * 0.15:
        return False
    cx = (x1 + x2) // 2
    if cx < fw * 0.1 or cx > fw * 0.9:
        return False
    ar = max(w, h) / max(min(w, h), 1)
    if ar > 2.0:
        return False
    return True


class ConeManager:
    def __init__(self):
        self.cones = []
        self.tap_counters = []
        self.tapped = []
        self.total_taps = 0

    def build(self, fw, fh):
        self.cones = []
        for (cx_f, cy_f, cw_f, ch_f) in CONE_DEFS:
            cx = int(cx_f * fw); cy = int(cy_f * fh)
            cw = int(cw_f * fw); ch = int(ch_f * fh)
            self.cones.append({
                "rect": (cx - cw//2, cy - ch//2, cx + cw//2, cy + ch//2),
                "center": (cx, cy),
            })
        self.reset_taps()

    def reset_taps(self):
        self.tap_counters = [0] * len(self.cones)
        self.tapped       = [False] * len(self.cones)
        self.total_taps   = 0

    def update(self, wrist_pts: list):
        newly_tapped = []
        for i, cone in enumerate(self.cones):
            x1, y1, x2, y2 = cone["rect"]
            inside = any(x1 <= wx <= x2 and y1 <= wy <= y2 for wx, wy in wrist_pts)
            if inside and not self.tapped[i]:
                self.tap_counters[i] += 1
                if self.tap_counters[i] >= CONE_TAP_HOLD_FRAMES:
                    self.tapped[i] = True
                    self.total_taps += 1
                    newly_tapped.append(i)
            elif not inside:
                self.tap_counters[i] = max(0, self.tap_counters[i] - 1)
        return newly_tapped

    def draw(self, frame):
        for i, cone in enumerate(self.cones):
            x1, y1, x2, y2 = cone["rect"]
            base_color = CONE_COLORS_BGR[i % len(CONE_COLORS_BGR)]
            if self.tapped[i]:
                color = (0, 255, 0)
                label = "TAPPED!"
            else:
                color = base_color
                label = f"Cone {i+1}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            pts = np.array([
                [(x1+x2)//2, y1+5],
                [x1+5, y2-5],
                [x2-5, y2-5],
            ], np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.putText(frame, label, (x1+4, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


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
    sets_per_leg = config.get("sets_per_leg", 3)
    hold_time = config.get("hold_time_seconds", 10)
    rest_time = config.get("rest_time_seconds", 45)
    min_bounces = config.get("min_bounces", 3)
    min_cone_taps = config.get("min_cone_taps", 2)
    frame_skip = config.get("frame_skip", 2)
    ball_conf_thresh = config.get("ball_conf_threshold", 0.70)
    max_ball_lost = config.get("max_ball_lost_frames", 3)
    bounce_line_ratio = config.get("bounce_line_ratio", 0.85)
    balance_thresh = config.get("balance_threshold", 0.05)
    smoothing_window = config.get("smoothing_window", 5)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    cone_mgr = ConeManager()
    smoother = AngleSmoother(window=smoothing_window)

    legs = ["left", "right"]
    leg_idx = 0
    sets_done = 0

    hold_start = None
    rest_start = None
    in_rest = False
    form_fail = False

    ball_center = None
    last_ball_center = None
    ball_lost_frames = 0
    frame_counter = 0
    last_ball_y = None
    ball_above_line = True
    bounce_count = 0

    form_violations = []
    form_bad_frames = 0
    MAX_FORM_BAD_FRAMES = 15

    total_sets_left = 0
    total_sets_right = 0
    data_buffer = []

    setup_mode = not args.headless

    log_event("READY", "One-Leg Obstacle Bonus active", {
        "sets_per_leg": sets_per_leg,
        "hold_time": hold_time,
        "min_bounces": min_bounces,
        "min_cone_taps": min_cone_taps
    })

    if setup_mode:
        tts_queue.put("One-Leg Obstacle Bonus exercise. Please complete the setup first.")

    try:
        with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    log_event("WARN", "No frame detected, ending loop")
                    break

                frame_counter += 1
                fh, fw = frame.shape[:2]
                bounce_line_y = int(fh * bounce_line_ratio)

                if not cone_mgr.cones:
                    cone_mgr.build(fw, fh)

                if setup_mode:
                    if not args.headless:
                        cv2.putText(frame, "ONE-LEG OBSTACLE BONUS - SETUP", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                        cv2.line(frame, (0, bounce_line_y), (fw, bounce_line_y), (0, 255, 255), 2)
                        cv2.putText(frame, "BOUNCE LINE", (fw-180, bounce_line_y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cv2.putText(frame, "Press 's' when ready", (20, fh-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cone_mgr.draw(frame)
                        cv2.imshow(args.game_id, frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('s'):
                            setup_mode = False
                            tts_queue.put(f"Setup complete. Balance on your {legs[leg_idx]} leg, bounce the ball, and tap the cones.")
                            time.sleep(1)
                        elif key == ord('q'):
                            stop_event.set()
                            break
                    continue

                if frame_counter % frame_skip == 0:
                    try:
                        results = model.track(frame, persist=True, verbose=False, conf=ball_conf_thresh)
                    except Exception as e:
                        log_event("ERROR", f"YOLO error: {e}")
                        break

                    current_ball = None
                    best_conf = 0.0
                    for r in results:
                        if r.boxes is None: continue
                        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                        clss = r.boxes.cls.cpu().numpy().astype(int)
                        confs = r.boxes.conf.cpu().numpy()
                        for box, cls, conf in zip(boxes, clss, confs):
                            if model.names[int(cls)] == "ball" and conf > ball_conf_thresh:
                                if is_valid_ball(box, fh, fw) and conf > best_conf:
                                    best_conf = conf
                                    x1,y1,x2,y2 = box
                                    current_ball = ((x1+x2)//2, (y1+y2)//2)
                    if current_ball:
                        ball_center = last_ball_center = current_ball
                        ball_lost_frames = 0
                    else:
                        ball_lost_frames += 1
                        ball_center = last_ball_center if ball_lost_frames <= max_ball_lost and last_ball_center else None

                if ball_center and not in_rest and hold_start is not None:
                    cx, cy = ball_center
                    if last_ball_y is not None:
                        if last_ball_y < bounce_line_y and cy >= bounce_line_y and ball_above_line:
                            bounce_count += 1
                            ball_above_line = False
                            log_event("BOUNCE", "Ball bounce detected", {"count": bounce_count})
                        elif cy < bounce_line_y:
                            ball_above_line = True
                    last_ball_y = cy

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(rgb)

                wrist_pts = []
                on_one_leg = False
                form_ok = False
                form_violations = []

                if pose_results.pose_landmarks:
                    if not args.headless:
                        mp_drawing.draw_landmarks(
                            frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(200,200,200), thickness=1, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(100,100,255), thickness=1),
                        )
                    lms = pose_results.pose_landmarks.landmark
                    PL = mp_pose.PoseLandmark

                    la = lms[PL.LEFT_ANKLE]
                    ra = lms[PL.RIGHT_ANKLE]
                    on_one_leg = (abs(la.y - ra.y) > balance_thresh or abs(la.visibility - ra.visibility) > 0.30)

                    for wrist_lm in [PL.LEFT_WRIST, PL.RIGHT_WRIST]:
                        lm = lms[wrist_lm]
                        if lm.visibility >= 0.65:
                            wrist_pts.append((int(lm.x * fw), int(lm.y * fh)))

                    la_score = la.y + la.visibility
                    ra_score = ra.y + ra.visibility
                    standing_side = "left" if la_score > ra_score else "right"

                    form_ok, form_violations = validate_pose(lms, fw, fh, standing_side, smoother, config)

                if in_rest:
                    elapsed_rest = time.time() - rest_start
                    remaining_rest = max(0, rest_time - elapsed_rest)

                    if not args.headless:
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
                        cv2.putText(frame, "REST", (fw//2 - 60, fh//2 - 60), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 4)
                        cv2.putText(frame, f"{int(remaining_rest)}s", (fw//2 - 40, fh//2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
                        cv2.putText(frame, f"Next: {legs[leg_idx].upper()} leg  |  Set {sets_done+1}/{sets_per_leg}", (fw//2 - 160, fh//2 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)
                        cone_mgr.draw(frame)
                        cv2.imshow(args.game_id, frame)
                    else:
                        time.sleep(0.01)

                    if elapsed_rest >= rest_time:
                        in_rest = False
                        form_fail = False
                        hold_start = None
                        bounce_count = 0
                        last_ball_y = None
                        ball_above_line = True
                        cone_mgr.reset_taps()
                        form_bad_frames = 0
                        tts_queue.put(f"Rest over. {legs[leg_idx].upper()} leg. Go!")
                        log_event("REST_END", f"Rest ended, starting {legs[leg_idx]} leg")

                    if not args.headless and cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_event.set()
                        break
                    continue

                current_leg = legs[leg_idx]

                if wrist_pts and hold_start is not None:
                    newly = cone_mgr.update(wrist_pts)
                    for ci in newly:
                        tts_queue.put(f"Cone {ci+1}!")
                        log_event("CONE_TAP", f"Cone {ci+1} tapped", {"total_taps": cone_mgr.total_taps})

                elapsed = time.time() - hold_start if hold_start else 0
                remaining = max(0.0, hold_time - elapsed)

                prereq_ok = on_one_leg and ball_center is not None

                if prereq_ok and not form_ok:
                    form_bad_frames += 1
                else:
                    form_bad_frames = max(0, form_bad_frames - 1)

                if hold_start is not None and form_bad_frames >= MAX_FORM_BAD_FRAMES:
                    tts_queue.put("Form broken. Resetting set. Fix your posture.")
                    log_event("FORM_FAIL", "Form broken, resetting", {"violations": form_violations})
                    hold_start = None
                    bounce_count = 0
                    last_ball_y = None
                    ball_above_line = True
                    cone_mgr.reset_taps()
                    form_bad_frames = 0

                if prereq_ok and form_ok:
                    if hold_start is None:
                        hold_start = time.time()
                        bounce_count = 0
                        last_ball_y = None
                        ball_above_line = True
                        cone_mgr.reset_taps()
                        elapsed = 0
                        remaining = float(hold_time)
                        tts_queue.put(f"{current_leg.upper()} leg. Bounce and tap cones. {hold_time} seconds.")
                        log_event("HOLD_START", "Starting active hold")

                    if elapsed >= hold_time:
                        ok_bounces = bounce_count >= min_bounces
                        ok_taps = cone_mgr.total_taps >= min_cone_taps

                        if ok_bounces and ok_taps:
                            if current_leg == "left":
                                total_sets_left += 1
                            else:
                                total_sets_right += 1
                            sets_done += 1
                            tts_queue.put(f"Set complete! {bounce_count} bounces, {cone_mgr.total_taps} cone taps.")
                            log_event("SET_COMPLETE", "Set completed successfully", {"bounces": bounce_count, "taps": cone_mgr.total_taps})

                            if sets_done >= sets_per_leg:
                                leg_idx = 1 - leg_idx
                                sets_done = 0
                                if leg_idx == 0:
                                    tts_queue.put("All sets complete for both legs! Outstanding work!")
                                    log_event("COMPLETED", "All sets complete")
                                    stop_event.set()
                                    break
                                else:
                                    tts_queue.put(f"Left leg done! Switching to {legs[leg_idx]} leg after rest.")

                            in_rest = True
                            rest_start = time.time()
                            hold_start = None
                            bounce_count = 0
                            cone_mgr.reset_taps()
                        else:
                            msgs = []
                            if not ok_bounces: msgs.append(f"only {bounce_count} bounces")
                            if not ok_taps: msgs.append(f"only {cone_mgr.total_taps} cone taps")
                            tts_queue.put("Incomplete: " + " and ".join(msgs) + ". Try again after rest.")
                            log_event("SET_FAILED", "Incomplete set", {"msgs": msgs})
                            form_fail = True
                            in_rest = True
                            rest_start = time.time()
                            hold_start = None
                            bounce_count = 0
                            cone_mgr.reset_taps()
                else:
                    if hold_start is not None:
                        hold_start = None
                        bounce_count = 0
                        last_ball_y = None

                data_buffer.append({
                    "timestamp": datetime.now().isoformat(),
                    "frame": frame_counter,
                    "leg": current_leg,
                    "set": sets_done,
                    "remaining_time": remaining,
                    "bounce_count": bounce_count,
                    "cone_taps": cone_mgr.total_taps,
                    "form_ok": form_ok,
                    "on_one_leg": on_one_leg,
                    "ball_visible": ball_center is not None
                })

                if not args.headless:
                    cone_mgr.draw(frame)
                    cv2.line(frame, (0, bounce_line_y), (fw, bounce_line_y), (0, 255, 255), 2)
                    cv2.putText(frame, "BOUNCE LINE", (fw-160, bounce_line_y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

                    if ball_center:
                        bcx, bcy = ball_center
                        r_size = 20 if bcy >= bounce_line_y else 12
                        b_col = (0, 255, 255) if bcy >= bounce_line_y else (0, 255, 0)
                        cv2.circle(frame, (bcx, bcy), r_size, b_col, -1)
                        cv2.circle(frame, (bcx, bcy), r_size+3, b_col, 2)

                    for wx, wy in wrist_pts:
                        cv2.circle(frame, (wx, wy), 8, (255, 0, 255), -1)

                    if not prereq_ok:
                        y_off = 170
                        if not on_one_leg:
                            cv2.putText(frame, "Stand on ONE leg", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            y_off += 28
                        if ball_center is None:
                            cv2.putText(frame, "Keep ball visible", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.rectangle(frame, (0, 0), (fw, 155), (0, 0, 0), -1)
                    cv2.rectangle(frame, (0, 0), (fw, 155), (50, 50, 50), 1)
                    cv2.putText(frame, f"ONE-LEG OBSTACLE  |  Leg: {current_leg.upper()}  |  Set: {sets_done+1}/{sets_per_leg}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

                    bar_w = int((fw - 20) * max(0, remaining) / hold_time)
                    bar_color = (0, 200, 0) if remaining > 4 else (0, 100, 255)
                    cv2.rectangle(frame, (10, 38), (fw-10, 58), (60, 60, 60), -1)
                    cv2.rectangle(frame, (10, 38), (10 + bar_w, 58), bar_color, -1)
                    cv2.putText(frame, f"{remaining:.1f}s", (fw//2 - 20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                    b_col = (0, 255, 0) if bounce_count >= min_bounces else (0, 165, 255)
                    t_col = (0, 255, 0) if cone_mgr.total_taps >= min_cone_taps else (0, 165, 255)
                    cv2.putText(frame, f"Bounces: {bounce_count}/{min_bounces}", (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, b_col, 2)
                    cv2.putText(frame, f"Cone Taps: {cone_mgr.total_taps}/{min_cone_taps}", (fw//2, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, t_col, 2)

                    if form_violations:
                        for j, v in enumerate(form_violations[:3]):
                            cv2.putText(frame, f"FORM: {v}", (10, 108 + j*20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 1)
                    else:
                        cv2.putText(frame, "FORM: OK", (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    cv2.imshow(args.game_id, frame)
                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('q'):
                        stop_event.set()
                        break
                    elif key == ord('r'):
                        tts_queue.put("Resetting session.")
                        leg_idx = 0; sets_done = 0
                        hold_start = None; rest_start = None
                        in_rest = False; form_fail = False
                        ball_center = None; last_ball_center = None
                        ball_lost_frames = 0; bounce_count = 0
                        last_ball_y = None; ball_above_line = True
                        cone_mgr.reset_taps(); form_bad_frames = 0
                        total_sets_left = 0; total_sets_right = 0
                else:
                    time.sleep(0.01)

    except Exception as e:
        log_event("ERROR", f"Vision loop exception: {str(e)}")
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()
        save_data(data_buffer, args.output, args.game_id, "oneleg_obstacle")
        log_event("CLEANUP", "Camera released", {
            "total_left": total_sets_left,
            "total_right": total_sets_right
        })


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
    parser = argparse.ArgumentParser(description="One-Leg Obstacle Bonus")
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
