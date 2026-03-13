import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import serial
import time

# ===========================
# Load YOLO model
# ===========================
model = YOLO("final.pt")

# ===========================
# Color map for each class
# (removed: couch, door / added: person)
# ===========================
class_colors = {
    "chair":  (0, 255, 0),
    "table":  (255, 0, 0),
    "person": (0, 165, 255),  # orange-ish (BGR)
}

# ===========================
# Confidence threshold for ESP32 signaling
# (Bounding boxes will still be drawn even if confidence < 0.65)
# ===========================
CONF_THRESHOLD = 0.65
CENTER_MARGIN_RATIO = 0.15  # 15% of image width treated as "center"

# ===========================
# ESP32 sending control
# ===========================
DIST_THRESHOLD = 2.0      # meters (considered "close" if < this)
SEND_INTERVAL = 0.1       # seconds
last_send_time = 0.0
last_sent_label = "none"  # to avoid repeating the same sound

# ===========================
# Serial connection to ESP32
# ===========================
try:
    ser = serial.Serial("COM6", 115200, timeout=0.1)
    time.sleep(2)
    print("Arduino connected")
except Exception:
    ser = None
    print("⚠ Arduino not connected")

# ===========================
# RealSense Setup
# ===========================
pipeline = rs.pipeline()
config = rs.config()

# Enable color and depth streams (640×480 @ 30FPS)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Align depth frame to the color frame
align = rs.align(rs.stream.color)

# Depth scale (to convert raw depth values to meters)
# NOTE:  depth_sensor.get_depth_scale() 
depth_scale = 0.0010000000474974513

print("Running... (Press 'q' to quit)")

try:
    while True:
        # ===== Get aligned frames =====
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data()) * depth_scale

        h, w, _ = color_image.shape
        image_center_x = w / 2
        center_margin = CENTER_MARGIN_RATIO * w

        # Run YOLO inference on the color image
        results = model(color_image, verbose=False)

        # Store only the close objects for ESP32 (chair/table/person)
        esp32_candidates = []   # (distance, label, side, confidence)

        # ===== Process each detected YOLO bounding box =====
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            label_name = model.names[cls_id]
            conf = float(box.conf[0])

            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)

            # Safety clamp (avoid out-of-range slicing)
            x1i = max(0, min(x1i, w - 1))
            x2i = max(0, min(x2i, w))
            y1i = max(0, min(y1i, h - 1))
            y2i = max(0, min(y2i, h))

            depth_region = depth_image[y1i:y2i, x1i:x2i]
            if depth_region.size == 0:
                continue

            obj_depth = np.median(depth_region)
            if np.isnan(obj_depth) or obj_depth <= 0:
                continue

            # ===== Draw bounding box regardless of depth =====
            color = class_colors.get(label_name, (255, 255, 255))
            cv2.rectangle(color_image, (x1i, y1i), (x2i, y2i), color, 2)

            label_text = f"{label_name}: {obj_depth:.2f}m  ({conf*100:.1f}%)"
            cv2.putText(
                color_image, label_text, (x1i, max(0, y1i - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )

            # ===== Add to ESP32 candidates only if:
            #   - object is near (< DIST_THRESHOLD)
            #   - and label is chair/table/person
            if obj_depth < DIST_THRESHOLD and label_name in ("chair", "table", "person"):
                bbox_center_x = (x1i + x2i) / 2

                if abs(bbox_center_x - image_center_x) < center_margin:
                    obstacle_side = "center"
                elif bbox_center_x < image_center_x:
                    obstacle_side = "left"
                else:
                    obstacle_side = "right"

                esp32_candidates.append((obj_depth, label_name, obstacle_side, conf))

        # ===== ESP32 SIGNALING SECTION (rate-limited + no repeat label) =====
        now = time.time()

        if ser and (now - last_send_time) >= SEND_INTERVAL:
            last_send_time = now

            send_label = "none"
            send_conf = 0.0
            send_dist = None
            send_side = None

            if esp32_candidates:
                esp32_candidates.sort(key=lambda x: x[0])
                min_dist, min_label, min_side, min_conf = esp32_candidates[0]

                send_dist = min_dist
                send_conf = min_conf
                send_side = min_side

                if min_conf >= CONF_THRESHOLD:
                    send_label = min_label
                else:
                    send_label = "none"

            # prevent repeating the same label
            if send_label == last_sent_label:
                if send_label == "none":
                    print("TX skip (still none)")
                else:
                    print(f"TX skip (still {send_label})")
            else:
                # Send to ESP32
                if send_label == "none":
                    ser.write(b"OBJ,none\n")
                else:
                    # left/right/center
                    ser.write(
                        f"OBJ,{send_label},{send_dist:.2f},{send_side}\n".encode("ascii")
                    )

                last_sent_label = send_label

                if send_label == "none":
                    if not esp32_candidates:
                        print(f"TX -> none (no chair/table/person < {DIST_THRESHOLD}m)")
                    else:
                        print(f"TX -> none (conf={send_conf:.2f} < {CONF_THRESHOLD})")
                else:
                    print(f"TX -> {send_label} (dist={send_dist:.2f}m, conf={send_conf:.2f}, side={send_side})")

        cv2.imshow("RealSense + YOLO11", color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Forced stop")

pipeline.stop()
cv2.destroyAllWindows()

if ser:
    ser.close()

print("Shutdown complete.")
