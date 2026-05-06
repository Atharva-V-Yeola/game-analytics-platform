#!/usr/bin/env python3
import cv2
import os
import platform
import sys

print(f"OS: {platform.system()}")
print(f"Python: {sys.executable}")
print(f"OpenCV version: {cv2.__version__}")
print(f"Display env: {os.environ.get('DISPLAY', 'NOT SET')}")

# Test 1: Camera
print("\n--- TEST 1: Camera ---")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("FAIL: cv2.VideoCapture(0) failed")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if cap.isOpened():
        print("OK: V4L2 backend works")
    else:
        print("FAIL: V4L2 also failed. Camera not accessible.")
        print("Fix: sudo usermod -a -G video $USER  (then logout/login)")
else:
    ret, frame = cap.read()
    if ret:
        print(f"OK: Camera opened, frame shape: {frame.shape}")
    else:
        print("FAIL: Camera opened but cannot read frames")
cap.release()

# Test 2: Display (GUI)
print("\n--- TEST 2: Display ---")
try:
    img = cv2.imread("/usr/share/pixmaps/ubuntu-logo.png")  # Any image
    if img is None:
        img = cv2.imread("/usr/share/icons/hicolor/48x48/apps/firefox.png")
    if img is None:
        img = cv2.imread("/usr/share/pixmaps/debian-logo.png")
    if img is None:
        # Create a blank image if no system image found
        img = cv2.imread("/dev/null")  # Will be None
        print("SKIP: No test image found, creating blank...")
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    cv2.imshow("Test Window", img)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    print("OK: Display/GUI works")
except Exception as e:
    print(f"FAIL: Cannot show window: {e}")
    print("Fix: You are on headless Ubuntu. Use --headless flag.")

# Test 3: Model file
print("\n--- TEST 3: Model file ---")
project_root = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(project_root, "models", "all_in_one_yolov82_best.pt")
print(f"Looking for: {model_path}")
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"OK: Model found ({size_mb:.1f} MB)")
else:
    print("FAIL: Model file not found")
    print(f"Fix: Place best.pt at: {model_path}")

# Test 4: Ultralytics
print("\n--- TEST 4: Ultralytics ---")
try:
    from ultralytics import YOLO
    print("OK: ultralytics installed")
except ImportError:
    print("FAIL: ultralytics not installed")
    print("Fix: pip install ultralytics")

# Test 5: MediaPipe
print("\n--- TEST 5: MediaPipe ---")
try:
    import mediapipe as mp
    print(f"OK: mediapipe {mp.__version__} installed")
    # Quick init test
    pose = mp.solutions.pose.Pose(static_image_mode=True)
    pose.close()
    print("OK: MediaPipe Pose initialized successfully")
except ImportError:
    print("FAIL: mediapipe not installed")
    print("Fix: pip install mediapipe")
except Exception as e:
    print(f"FAIL: MediaPipe initialization failed: {e}")

print("\n--- END ---")