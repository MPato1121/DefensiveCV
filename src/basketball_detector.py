
import cv2
import torch
from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess
import numpy as np
import os
import time

class BasketballDetector:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.exp = get_exp(None, "yolox-s")
        self.model = self.exp.get_model()
        self.model.to(self.device)
        self.model.eval()
        
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])

        self.preproc = ValTransform(legacy=False)
        
        print("Model loaded successfully.")

    def detect(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"Failed to open video: {video_path}"

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Using XVID for better compatibility
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        positions = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            img, ratio = self.preproc(frame, None, self.exp.test_size)
            img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                outputs = self.model(img)
                outputs = postprocess(outputs, num_classes=self.exp.num_classes, conf_thre=0.3, nms_thre=0.65)

            if outputs[0] is not None:
                outputs = outputs[0].cpu().numpy()
                for output in outputs:
                    x0, y0, x1, y1, conf, cls_id = output
                    x0, y0, x1, y1 = int(x0 / ratio), int(y0 / ratio), int(x1 / ratio), int(y1 / ratio)

                    # Save position data
                    positions.append({
                        "frame": cap.get(cv2.CAP_PROP_POS_FRAMES),
                        "cls_id": int(cls_id),
                        "confidence": float(conf),
                        "bbox": [x0, y0, x1, y1]
                    })

                    label = f"{int(cls_id)}: {conf:.2f}"
                    color = (0, 255, 0) if int(cls_id) == 0 else (255, 0, 0)  # Green for players, Blue for ball
                    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                    cv2.putText(frame, label, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            out.write(frame)

        cap.release()
        out.release()
        print(f"Processed video saved to: {output_path}")

        return positions

if __name__ == "__main__":
    model_path = "./models/yolox_s.pth"  # Path to the YOLOX model weights
    video_path = "./data/videos/sample.mp4"  # Path to input video
    output_path = "./data/videos/output.mp4"  # Path to save output

    detector = BasketballDetector(model_path=model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    positions = detector.detect(video_path, output_path)

    # Save positions to a JSON file for analysis
    import json
    with open("./data/positions.json", "w") as f:
        json.dump(positions, f, indent=4)

    print(f"Saved object positions to ./data/positions.json")
