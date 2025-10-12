import cv2
from ultralytics import YOLO
from collections import defaultdict
import time

model = YOLO("yolo11s.pt")
model.fuse() 

if model.device.type != "cpu":
    model.model.half()

line_y_red = 200
class_counts = defaultdict(int)
crossed_ids = set()
prev_positions = {}  

results = model.track(
    source="experiments/cv-yolo/yolo_data.mp4",
    persist=True,
    stream=True,
    classes=[1, 2, 3, 5, 6, 7],   
    tracker="botsort.yaml",  
)

frame_count = 0
prev_time = time.time()

for result in results:
    frame = result.orig_img
    if frame is None or result.boxes is None:
        continue

    frame_count += 1
    h, w = frame.shape[:2]

    boxes = result.boxes.xyxy
    ids = result.boxes.id
    clss = result.boxes.cls
    confs = result.boxes.conf

    if boxes is None or ids is None:
        continue

    boxes = boxes.cpu().numpy()
    ids = ids.int().cpu().tolist()
    clss = clss.int().cpu().tolist()
    confs = confs.cpu().numpy()

    for box, track_id, class_idx, conf in zip(boxes, ids, clss, confs):
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        class_name = model.names[class_idx]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, f"ID:{track_id} {class_name}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if frame_count < 30:
            continue

        prev_y = prev_positions.get(track_id, cy)
        prev_positions[track_id] = cy

        if prev_y < line_y_red <= cy and track_id not in crossed_ids:
            crossed_ids.add(track_id)
            class_counts[class_name] += 1

    cv2.line(frame, (0, line_y_red), (w, line_y_red), (0, 0, 255), 3)

    y_offset = 30
    for cname, count in class_counts.items():
        cv2.putText(frame, f"{cname}: {count}", (50, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += 30

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (50, y_offset + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("YOLO Object Tracking & Counting (Fast & Accurate)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord('q')]:
        print("Exiting...")
        break

cv2.destroyAllWindows()
