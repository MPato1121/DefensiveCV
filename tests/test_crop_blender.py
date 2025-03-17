import cv2
import numpy as np
from src.crop_blender import CropBlender

def test_crop_blender():
    blender = CropBlender(buffer_size=3)

    # Create mock crops (3 crops of size 50x50x3)
    crop1 = np.ones((50, 50, 3), dtype=np.uint8) * 50
    crop2 = np.ones((50, 50, 3), dtype=np.uint8) * 100
    crop3 = np.ones((50, 50, 3), dtype=np.uint8) * 150

    blender.add_crop(1, crop1)
    blender.add_crop(1, crop2)
    blender.add_crop(1, crop3)

    blended = blender.get_blended_crop(1)

    assert blended.shape == (50, 50, 3), "Blended crop shape mismatch"
    assert blended.mean() == 100, f"Blended crop mean is {blended.mean()} instead of 100"

    print("✅ Crop blending test passed!")

if __name__ == "__main__":
    test_crop_blender()
