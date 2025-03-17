import cv2
import torch
import numpy as np
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
import os

class BasketballDetector:
    def __init__(self, model_path, device):
        self.device = device
        self.model = YOLO(model_path).to(device)

        # Tracker configuration
        class TrackerArgs:
            track_thresh = 0.5
            track_buffer = 30
            match_thresh = 0.8
            aspect_ratio_thresh = 1.6
            min_box_area = 10
            mot20 = False

        args = TrackerArgs()
        self.tracker = BYTETracker(args)

    def detect(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0,
                              (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLOv10 detection
            results = self.model(frame)

            # Convert YOLOv10 output to ByteTrack format
            detections = []
            for r in results:
                for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                    x1, y1, x2, y2 = box
                    detections.append([float(x1), float(y1), float(x2), float(y2), float(conf), int(cls)])

            detections = np.array(detections)
            if len(detections) > 0:
                detections = torch.from_numpy(detections).float().cpu()

                # Update tracker
                tracked_objects = self.tracker.update(detections, frame.shape[:2], (frame.shape[0], frame.shape[1]))

                for track in tracked_objects:
                    x1, y1, w, h = track.tlwh
                    x2, y2 = x1 + w, y1 + h
                    track_id = track.track_id
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            out.write(frame)

        cap.release()
        out.release()

        print(f"✅ Detection complete. Results saved to {output_path}")
