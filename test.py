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
    # Check if mp.solutions is available (may need shim for 0.10.14+)
    if not hasattr(mp, 'solutions'):
        import mediapipe.python.solutions as _solutions
        mp.solutions = _solutions
        print("INFO: Applied mp.solutions compatibility shim (mediapipe 0.10.14+)")
    # Quick init test
    pose = mp.solutions.pose.Pose(static_image_mode=True)
    pose.close()
    print("OK: MediaPipe Pose initialized successfully")
except ImportError:
    print("FAIL: mediapipe not installed")
    print("Fix: pip install mediapipe")
except Exception as e:
    print(f"FAIL: MediaPipe initialization failed: {e}")

# Test 6: Protobuf compatibility
print("\n--- TEST 6: Protobuf ---")
try:
    import google.protobuf
    ver = google.protobuf.__version__
    major = int(ver.split('.')[0])
    if major >= 5:
        print(f"WARN: protobuf {ver} may conflict with mediapipe (needs <5)")
        print("Fix: pip install 'protobuf>=4.25.3,<5'")
    else:
        print(f"OK: protobuf {ver} (compatible with mediapipe)")
except ImportError:
    print("FAIL: protobuf not installed")

# Test 7: Windows COM support (pywin32)
print("\n--- TEST 7: Windows COM (pywin32) ---")
if platform.system() == "Windows":
    try:
        import pythoncom
        print("OK: pywin32/pythoncom available (COM-safe TTS enabled)")
    except ImportError:
        print("WARN: pywin32 not installed — TTS may freeze with MediaPipe games")
        print("Fix: pip install pywin32")
else:
    print("SKIP: Not on Windows (COM not needed)")

# Test 8: pyttsx3 TTS
print("\n--- TEST 8: pyttsx3 TTS ---")
try:
    import pyttsx3
    engine = pyttsx3.init()
    engine.stop()
    del engine
    print("OK: pyttsx3 initialized successfully")
except ImportError:
    print("WARN: pyttsx3 not installed — TTS coaching will be disabled")
    print("Fix: pip install pyttsx3")
except Exception as e:
    print(f"WARN: pyttsx3 init failed: {e}")
    print("Fix: TTS coaching will be disabled but games will still work")

print("\n--- END ---")