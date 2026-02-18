# test_camera.py
import cv2
import numpy as np
import time
import json
import os
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import load_model

# ========================
# 1. SETTINGS
# ========================
IMG_SIZE = (224, 224)
MODEL_PATH = 'hospital_gesture_model_final.keras'
CLASS_NAMES_PATH = 'class_names.json'
CONFIG_DIR = 'config'
MAPPING_PATH = os.path.join(CONFIG_DIR, 'medical_gesture_mapping.json')
CONFIDENCE_THRESHOLD = 0.55
DISPLAY_DURATION = 10  # ثواني عرض الاشارة

# ========================
# 2. LOAD CLASS NAMES
# ========================
class_names = []
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    print(f"Loaded {len(class_names)} class names")
    print(f"Classes from model: {class_names}")  # DEBUG
else:
    print(f"Error: Class names file not found at {CLASS_NAMES_PATH}")
    exit()

# ========================
# 3. LOAD MODEL
# ========================
print("=" * 60)
print("Loading model...")

try:
    model = load_model(MODEL_PATH, compile=False)
    print(f"Model loaded: {MODEL_PATH}")
except Exception as e1:
    print(f"Trying alternative loading method...")
    try:
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense,
                                             Dropout, BatchNormalization, Activation)
        from tensorflow.keras.regularizers import l2

        NUM_CLASSES = len(class_names)
        base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, kernel_regularizer=l2(0.0001)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Dense(256, kernel_regularizer=l2(0.0001)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Dense(NUM_CLASSES, activation='softmax')
        ])
        model.load_weights(MODEL_PATH)
        print(f"Model weights loaded! Classes: {NUM_CLASSES}")
    except Exception as e2:
        print(f"Error: {e2}")
        exit()

# ========================
# 4. LOAD MEDICAL GESTURE MAPPING
# ========================
MEDICAL_GESTURE_MAP = {}

if os.path.exists(MAPPING_PATH):
    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        MEDICAL_GESTURE_MAP = json.load(f)
    print(f"Loaded {len(MEDICAL_GESTURE_MAP)} gesture mappings")
    print(f"Mapped gestures: {list(MEDICAL_GESTURE_MAP.keys())}")  # DEBUG
else:
    print(f"Warning: Mapping file not found at {MAPPING_PATH}")

# ========================
# 5. HELPER FUNCTIONS
# ========================
def clean_gesture_name(gesture):
    """Remove all known prefixes"""
    clean = gesture
    for prefix in ['train_val_', 'test_val_', 'train_', 'test_', 'val_']:
        clean = clean.replace(prefix, '')
    return clean.strip().lower()


def get_medical_info(gesture_name):
    clean = clean_gesture_name(gesture_name)
    print(f"DEBUG -> raw: '{gesture_name}' | clean: '{clean}'")  # DEBUG

    if clean in MEDICAL_GESTURE_MAP:
        info = MEDICAL_GESTURE_MAP[clean]
        return {
            'found': True,
            'raw_name': gesture_name,
            'clean_name': clean,
            'action': info.get('action', 'N/A'),
            'description': info.get('description', 'No description'),
            'medical_use': info.get('medical_use', 'N/A'),
            'priority': info.get('priority', 'low'),
            'alert_level': info.get('alert_level', None)
        }
    return {
        'found': False,
        'raw_name': gesture_name,
        'clean_name': clean,
        'action': 'Unknown',
        'description': 'Not mapped',
        'medical_use': 'N/A',
        'priority': 'low',
        'alert_level': None
    }


def preprocess_frame(frame):
    h, w = frame.shape[:2]
    size = min(h, w)
    start_h = (h - size) // 2
    start_w = (w - size) // 2
    roi = frame[start_h:start_h + size, start_w:start_w + size]
    resized = cv2.resize(roi, IMG_SIZE)
    normalized = resized / 255.0
    input_tensor = np.expand_dims(normalized, axis=0).astype(np.float32)
    return input_tensor, (start_w, start_h, size)


def get_priority_color(priority):
    colors = {
        'critical': (0, 0, 255),
        'high':     (0, 100, 255),
        'medium':   (0, 255, 255),
        'low':      (0, 255, 0),
    }
    return colors.get(priority, (180, 180, 180))


def draw_text_with_bg(frame, text, pos, font_scale=0.7, color=(255,255,255), thickness=2):
    """Draw text with dark background for readability"""
    x, y = pos
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(frame, (x-2, y-th-4), (x+tw+2, y+4), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


# ========================
# 6. OPEN CAMERA
# ========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open camera!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("=" * 60)
print("Camera is running!")
print("Press 'q' to quit")
print("=" * 60)

# ========================
# 7. TRACKING VARIABLES
# ========================
prev_time = time.time()
last_pred_time = 0
PRED_INTERVAL = 0.3

# اخر اشاره اتلقطت
last_gesture_info = None
gesture_display_start = 0  # وقت ما اتلقطت الاشاره
current_conf = 0.0
action_log = []

# ========================
# 8. MAIN LOOP
# ========================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        current_time = time.time()

        # ---- PREDICTION ----
        if current_time - last_pred_time >= PRED_INTERVAL:
            input_tensor, _ = preprocess_frame(frame)
            predictions = model.predict(input_tensor, verbose=0)[0]
            idx = np.argmax(predictions)
            conf = float(predictions[idx])
            gesture_raw = class_names[idx]
            current_conf = conf

            if conf >= CONFIDENCE_THRESHOLD:
                info = get_medical_info(gesture_raw)

                # لو اشاره جديده مختلفة عن اللي قبلها
                if (last_gesture_info is None or
                        info['clean_name'] != last_gesture_info['clean_name']):
                    last_gesture_info = info
                    gesture_display_start = current_time

                    # Log
                    action_log.append({
                        'time': datetime.now().isoformat(),
                        'gesture': info['clean_name'],
                        'found_in_map': info['found'],
                        'action': info['action'],
                        'confidence': round(conf, 3)
                    })

            last_pred_time = current_time

        # ---- FPS ----
        fps = 1.0 / max(current_time - prev_time, 1e-9)
        prev_time = current_time

        # ---- وقت العرض المتبقي ----
        time_since_gesture = current_time - gesture_display_start
        still_displaying = (last_gesture_info is not None and
                           time_since_gesture <= DISPLAY_DURATION)

        # ========================
        # 9. DRAW ROI BOX
        # ========================
        _, (roi_x, roi_y, roi_size) = preprocess_frame(frame)
        cv2.rectangle(frame,
                      (roi_x, roi_y),
                      (roi_x + roi_size, roi_y + roi_size),
                      (0, 255, 0), 2)

        # ========================
        # 10. TOP INFO BAR
        # ========================
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (15, 15, 15), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

        # Confidence bar
        bar_w = int(current_conf * 180)
        bar_color = (0, 255, 0) if current_conf > 0.75 else \
                    (0, 200, 255) if current_conf > 0.55 else (0, 0, 255)
        cv2.rectangle(frame, (80, 8), (270, 22), (50, 50, 50), -1)
        cv2.rectangle(frame, (80, 8), (80 + bar_w, 22), bar_color, -1)
        cv2.putText(frame, f"Conf: {current_conf*100:.1f}%",
                    (275, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ========================
        # 11. GESTURE DISPLAY
        # ========================
        if still_displaying and last_gesture_info:
            info = last_gesture_info

            # Timer bar - بيقل مع الوقت
            timer_ratio = max(0, 1 - (time_since_gesture / DISPLAY_DURATION))
            timer_w = int(w * timer_ratio)
            timer_color = (0, 255, 0) if timer_ratio > 0.5 else \
                          (0, 200, 255) if timer_ratio > 0.25 else (0, 0, 255)
            cv2.rectangle(frame, (0, h-8), (w, h), (30, 30, 30), -1)
            cv2.rectangle(frame, (0, h-8), (timer_w, h), timer_color, -1)

            # ========================
            # الاشاره ليها معني طبي
            # ========================
            if info['found']:
                priority = info['priority']
                p_color = get_priority_color(priority)

                # Panel background
                overlay2 = frame.copy()
                cv2.rectangle(overlay2, (0, 55), (w, 260), (10, 10, 30), -1)
                frame = cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0)

                # Action - اكبر نص
                action_text = info['action'].replace('_', ' ').upper()
                cv2.putText(frame, action_text,
                            (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, p_color, 2)

                # Description
                cv2.putText(frame, info['description'],
                            (10, 135),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Medical use
                cv2.putText(frame, f"Medical: {info['medical_use']}",
                            (10, 165),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                # Priority
                cv2.putText(frame, f"Priority: {priority.upper()}",
                            (10, 198),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, p_color, 2)

                # Timer text
                secs_left = int(DISPLAY_DURATION - time_since_gesture) + 1
                cv2.putText(frame, f"({secs_left}s)",
                            (w - 70, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

                # Emergency effects
                if info['alert_level'] == 'red':
                    cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 6)
                    cv2.putText(frame, "!!! EMERGENCY !!!",
                                (w//2 - 160, h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                elif info['alert_level'] == 'orange':
                    cv2.rectangle(frame, (0, 0), (w, h), (0, 100, 255), 4)

            # ========================
            # الاشاره مش في الـ mapping
            # ========================
            else:
                overlay2 = frame.copy()
                cv2.rectangle(overlay2, (0, 55), (w, 180), (20, 10, 10), -1)
                frame = cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0)

                # Class name
                cv2.putText(frame,
                            f"Class: {info['clean_name']}",
                            (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 255), 2)

                cv2.putText(frame,
                            "No medical mapping assigned",
                            (10, 135),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

                secs_left = int(DISPLAY_DURATION - time_since_gesture) + 1
                cv2.putText(frame, f"({secs_left}s)",
                            (w - 70, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        else:
            # مفيش اشاره او انتهى الوقت
            cv2.putText(frame,
                        "Show your hand...",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 1)

        # Footer
        cv2.putText(frame, "Press 'q' to quit",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

        # ========================
        # 12. SHOW FRAME
        # ========================
        cv2.imshow("Hospital Gesture Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("Session Summary")
    print(f"Total gestures detected: {len(action_log)}")

    if action_log:
        from collections import Counter
        counts = Counter(e['gesture'] for e in action_log)
        most_common = counts.most_common(1)[0]
        print(f"Most used: {most_common[0]} ({most_common[1]} times)")

        print("\nAll detected gestures:")
        for entry in action_log:
            print(f"  {entry['time']} | {entry['gesture']} | {entry['action']} | {entry['confidence']}")

        log_file = f"gesture_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(action_log, f, indent=2)
        print(f"\nLog saved: {log_file}")

    print("Camera closed successfully")
    print("=" * 60)
