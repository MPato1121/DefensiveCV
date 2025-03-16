import cv2
import torch
from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess
import numpy as np
import os
import time
import json
import argparse

class BasketballDetector:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.exp = get_exp(None, "yolox-s")
        self.model = self.exp.get_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Load model checkpoint with error handling
        try:
            ckpt = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            print("Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model checkpoint: {e}")

        self.preproc = ValTransform(legacy=False)
        
        # Class labels for players and ball
        self.class_labels = {0: "player", 1: "ball"}

    def detect(self, video_path, output_path, frame_skip=1):
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"Failed to open video: {video_path}"

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        positions = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            # Preprocessing step
            img, ratio = self.preproc(frame, None, self.exp.test_size)

            # Catch invalid ratio issue
            if ratio == 0 or ratio is None:
                raise ValueError("Invalid ratio value returned by preprocessing.")

            img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                outputs = self.model(img)
                outputs = postprocess(outputs, num_classes=self.exp.num_classes, conf_thre=0.3, nms_thre=0.65)

            if outputs[0] is not None and len(outputs[0].shape) == 2:
                for output in outputs[0]:
                    if len(output) < 6:
                        continue

                    # Convert to scalar values
                    x0, y0, x1, y1, conf, cls_id = output[:6]
                    x0 = int(float(x0.item()) / ratio)
                    y0 = int(float(y0.item()) / ratio)
                    x1 = int(float(x1.item()) / ratio)
                    y1 = int(float(y1.item()) / ratio)

                    positions.append({
                        "frame": frame_count,
                        "cls_id": int(cls_id.item()),
                        "class_label": self.class_labels.get(int(cls_id.item()), "unknown"),
                        "confidence": float(conf.item()),
                        "bbox": [x0, y0, x1, y1]
                    })

                    # Draw bounding box and label
                    label = f"{self.class_labels.get(int(cls_id.item()), 'unknown')}: {conf.item():.2f}"
                    color = (0, 255, 0) if int(cls_id.item()) == 0 else (255, 0, 0)
                    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                    cv2.putText(frame, label, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                out.write(frame)

        cap.release()
        out.release()
        print(f"Processed video saved to: {output_path}")

        return positions


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./models/yolox_s.pth", help="Path to YOLOX model weights")
    parser.add_argument("--video_path", default="./data/videos/sample.mp4", help="Path to input video")
    parser.add_argument("--output_path", default="./data/videos/output.mp4", help="Path to save output video")
    parser.add_argument("--positions_path", default="./data/positions.json", help="Path to save positions JSON")
    parser.add_argument("--frame_skip", type=int, default=1, help="Process every Nth frame (default: 1)")
    args = parser.parse_args()

    # Initialize detector and process video
    detector = BasketballDetector(model_path=args.model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    positions = detector.detect(args.video_path, args.output_path, frame_skip=args.frame_skip)

    # Save positions to JSON file
    with open(args.positions_path, "w") as f:
        json.dump(positions, f, indent=4)

    print(f"Saved object positions to {args.positions_path}")