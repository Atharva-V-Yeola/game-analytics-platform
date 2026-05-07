#!/usr/bin/env python3
"""
Quick test: Verifies pyttsx3 + MediaPipe can run together on Windows.
Run this on your Windows machine BEFORE testing the full games.

Usage:  venv\Scripts\python test_windows_tts.py
"""
import platform
import queue
import threading
import time
import sys

print(f"OS: {platform.system()}")
print(f"Python: {sys.version}")

# --- Test 1: pyttsx3 init ---
print("\n[TEST 1] pyttsx3 init...")
try:
    import pyttsx3
    if platform.system() == "Windows":
        import pythoncom
    print("  OK: imports work")
except ImportError as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# --- Test 2: MediaPipe init ---
print("\n[TEST 2] MediaPipe Pose init...")
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    pose.close()
    print("  OK: MediaPipe Pose initialized and closed")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# --- Test 3: TTS on dedicated thread with COM init (the actual fix) ---
print("\n[TEST 3] TTS worker thread + MediaPipe running together...")

speech_queue = queue.Queue()
stop_event = threading.Event()
tts_ok = threading.Event()

def speech_worker():
    # This is the EXACT pattern used in the game scripts
    if platform.system() == "Windows":
        pythoncom.CoInitialize()

    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.setProperty('volume', 1.0)

    # Signal that TTS initialized successfully
    tts_ok.set()

    while not stop_event.is_set():
        try:
            msg = speech_queue.get(timeout=0.5)
            if msg is None:
                break
            engine.say(msg)
            engine.runAndWait()
        except queue.Empty:
            continue

    if platform.system() == "Windows":
        pythoncom.CoUninitialize()

# Start TTS thread
t = threading.Thread(target=speech_worker, daemon=True)
t.start()

# Wait for TTS to initialize
if not tts_ok.wait(timeout=5):
    print("  FAIL: TTS thread did not initialize within 5 seconds (DEADLOCK)")
    sys.exit(1)

print("  OK: TTS engine initialized on worker thread")

# --- Test 4: Run MediaPipe on main thread while TTS speaks ---
print("\n[TEST 4] MediaPipe inference + TTS speaking simultaneously...")
import cv2
import numpy as np

# Create a dummy frame (no camera needed)
dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
rgb = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB)

speech_queue.put("Testing speech")

with mp_pose.Pose(min_detection_confidence=0.5) as pose:
    for i in range(10):
        results = pose.process(rgb)
        time.sleep(0.1)

print("  OK: MediaPipe processed 10 frames while TTS was active")

# Cleanup
speech_queue.put(None)
t.join(timeout=3)

print("\n=== ALL TESTS PASSED ===")
print("The pyttsx3 + MediaPipe fix is working correctly on this machine.")
