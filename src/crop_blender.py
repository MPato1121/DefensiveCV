import cv2
import numpy as np
from collections import defaultdict

class CropBlender:
    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.crop_buffer = defaultdict(list)
    
    def _compute_blur(self, crop):
        # Use variance of Laplacian to estimate blur
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _compute_brightness(self, crop):
        # Compute average brightness in grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)

    def _compute_histogram_similarity(self, crop1, crop2):
        hist1 = cv2.calcHist([crop1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([crop2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    def add_crop(self, track_id, crop):
        if len(self.crop_buffer[track_id]) >= self.buffer_size:
            self.crop_buffer[track_id].pop(0)
        
        # Filter out blurry or dark crops
        if self._compute_blur(crop) > 50 and self._compute_brightness(crop) > 30:
            if len(self.crop_buffer[track_id]) > 0:
                # Keep only if histogram similarity is high enough
                similarity = self._compute_histogram_similarity(self.crop_buffer[track_id][-1], crop)
                if similarity > 0.7:
                    self.crop_buffer[track_id].append(crop)
            else:
                self.crop_buffer[track_id].append(crop)

    def get_blended_crop(self, track_id):
        if len(self.crop_buffer[track_id]) == 0:
            return None
        
        # Weighted average: prioritize most recent crops
        weights = np.linspace(1, 2, len(self.crop_buffer[track_id]))
        weighted_crops = np.array(self.crop_buffer[track_id]) * weights[:, None, None, None]
        blended_crop = np.sum(weighted_crops, axis=0) / np.sum(weights)

        return blended_crop.astype(np.uint8)

    def clear(self):
        self.crop_buffer.clear()
