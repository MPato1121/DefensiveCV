import cv2
import torch
import numpy as np
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from preprocessor import Preprocessor
from postprocessor import Postprocessor
from clustering import Clustering
from court_mapper import CourtMapper
from vision_api import VisionAPI

class BasketballDetector_V2:
    def __init__(self, model_path, device, team_colors):
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

        # New modules
        self.preprocessor = Preprocessor()
        self.postprocessor = Postprocessor()
        self.clustering = Clustering()
        self.court_mapper = CourtMapper()
        self.vision_api = VisionAPI()
        
        # Dynamic team handling
        self.team_colors = team_colors
        self.classes = list(team_colors) + ["none"]

    def detect(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0,
                              (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # ✅ Step 1: Pre-process frame
            frame = self.preprocessor.preprocess(frame)

            # ✅ Step 2: YOLO Detection
            results = self.model(frame)

            # ✅ Step 3: Convert YOLO output to ByteTrack format
            detections = []
            for r in results:
                for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                    x1, y1, x2, y2 = box
                    detections.append([float(x1), float(y1), float(x2), float(y2), float(conf), int(cls)])

            detections = np.array(detections)
            if len(detections) > 0:
                detections = torch.from_numpy(detections).float().cpu()

                # ✅ Step 4: Post-processing to refine detections
                detections = self.postprocessor.filter_detections(detections)

                # ✅ Step 5: Update tracker
                tracked_objects = self.tracker.update(detections, frame.shape[:2], (frame.shape[0], frame.shape[1]))

                players = []
                ball = []
                for track in tracked_objects:
                    x1, y1, w, h = track.tlwh
                    x2, y2 = x1 + w, y1 + h
                    track_id = track.track_id

                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Separate players and ball
                    if track.class_id == 0:  # Player
                        players.append([x1, y1, x2, y2, track_id])
                    elif track.class_id == 31:  # Ball
                        ball.append([x1, y1, x2, y2, track_id])

                if len(players) > 0:
                    players = np.array(players)

                    # ✅ Step 6: Run ChatGPT Vision API for team classification
                    response = self.vision_api.process_batches(frame, players, self.team_colors)
                    if response:
                        players = self.vision_api.update_class_id(response, players, self.classes)

                    # ✅ Step 7: Clustering for ID stabilization (optional)
                    team_ids = self.clustering.kmeans_cluster(players[:, :4], k=2)

                    for i, team_id in enumerate(team_ids):
                        color = (0, 255, 0) if team_id == 0 else (0, 0, 255)
                        x1, y1, x2, y2 = players[i][:4]
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(frame, f'Team: {self.classes[team_id]}', (int(x1), int(y1) - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # ✅ Step 8: Court Mapping (optional)
                lines = self.court_mapper.detect_lines(frame)
                if lines is not None:
                    for line in lines:
                        rho, theta = line[0]
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # ✅ Step 9: Write frame to output
            out.write(frame)

        cap.release()
        out.release()
        print(f"✅ Detection complete. Results saved to {output_path}")

# Example usage:
if __name__ == "__main__":
    # Example dynamic team colors
    team_colors = ("white", "neon green")

    detector = BasketballDetector_V2(
        model_path="yolov10m.pt",
        device="cuda",
        team_colors=team_colors
    )
    detector.detect(
        video_path="input.mp4",
        output_path="output.mp4"
    )
