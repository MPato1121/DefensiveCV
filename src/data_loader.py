import os
import cv2
import pandas as pd
from google.colab.patches import cv2_imshow  # Add this import for Colab

class DataLoader:
    def __init__(self, video_dir='data/videos', box_score_dir='data/box_scores', season_stats_dir='data/season_stats'):
        self.video_dir = video_dir
        self.box_score_dir = box_score_dir
        self.season_stats_dir = season_stats_dir
        self.videos = {}  # Store loaded videos
        self.box_scores = {}  # Store loaded box scores
        self.season_stats = {}  # Store loaded season stats

    def load_videos(self):
        video_files = [f for f in os.listdir(self.video_dir) if f.endswith('.mp4')]
        if not video_files:
            print("[DEBUG] No video files found.")
            return {}
        
        for file in video_files:
            path = os.path.join(self.video_dir, file)
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"[DEBUG] Failed to open video file: {file}")
                continue
            
            self.videos[file] = cap
            print(f"[DEBUG] Loaded video: {file} | Total Frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

        return self.videos

    def load_box_scores(self):
        box_score_files = [f for f in os.listdir(self.box_score_dir) if f.endswith('.csv')]
        if not box_score_files:
            print("[DEBUG] No box score files found.")
            return {}
        
        for file in box_score_files:
            path = os.path.join(self.box_score_dir, file)
            try:
                df = pd.read_csv(path)
                self.box_scores[file] = df
                print(f"[DEBUG] Loaded box score file: {file} | Rows: {df.shape[0]}")
            except Exception as e:
                print(f"[DEBUG] Failed to load box score file {file}: {e}")

        return self.box_scores

    def load_season_stats(self):
        season_stat_files = [f for f in os.listdir(self.season_stats_dir) if f.endswith('.csv')]
        if not season_stat_files:
            print("[DEBUG] No season stat files found.")
            return {}
        
        for file in season_stat_files:
            path = os.path.join(self.season_stats_dir, file)
            try:
                df = pd.read_csv(path)
                self.season_stats[file] = df
                print(f"[DEBUG] Loaded season stat file: {file} | Rows: {df.shape[0]}")
            except Exception as e:
                print(f"[DEBUG] Failed to load season stat file {file}: {e}")

        return self.season_stats
    
    def show_video_frame(self, video_name, frame_number=0):
        if video_name not in self.videos:
            print(f"[DEBUG] Video '{video_name}' not loaded.")
            return
        
        cap = self.videos[video_name]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            # Use cv2_imshow instead of cv2.imshow for Colab
            cv2_imshow(frame)
        else:
            print(f"[DEBUG] Could not retrieve frame {frame_number} from {video_name}")

# Example usage
if __name__ == "__main__":
    print("[DEBUG] Starting data loading...")
    loader = DataLoader()
    videos = loader.load_videos()
    box_scores = loader.load_box_scores()
    season_stats = loader.load_season_stats()

    if videos:
        first_video = list(videos.keys())[0]
        print(f"[DEBUG] Attempting to show frame from {first_video}")
        loader.show_video_frame(first_video, frame_number=10)