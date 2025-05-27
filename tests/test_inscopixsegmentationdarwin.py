#!/usr/bin/env python
# test_with_real_data.py

import os
import sys
from .setup_paths import OPHYS_DATA_PATH
import platform

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roiextractors import InscopixSegmentationExtractor

# Path to your actual .isxd file
FILE_PATH = OPHYS_DATA_PATH / "segmentation_datasets" / "inscopix" / "cellset.isxd"  # Replace with your file path


def test_with_real_data():
    print(f"Running on: {platform.system()} {platform.machine()}")
    print(f"Testing with file: {FILE_PATH}")

    # Verify the file exists
    if not os.path.exists(FILE_PATH):
        print(f"ERROR: File does not exist: {FILE_PATH}")
        return False

    try:
        # Create the extractor
        extractor = InscopixSegmentationExtractor(file_path=FILE_PATH)

        # Basic checks
        num_rois = extractor.get_num_rois()
        print(f"Number of ROIs: {num_rois}")

        if num_rois > 0:
            roi_ids = extractor.get_roi_ids()
            print(f"ROI IDs: {roi_ids[:5]}{'...' if len(roi_ids) > 5 else ''}")

            # Get traces
            traces = extractor.get_traces()
            print(f"Traces shape: {traces.shape}")

            # Get image size
            img_size = extractor.get_image_size()
            print(f"Image size: {img_size}")

        print("All operations completed successfully!")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_with_real_data()
    print(f"Test {'passed' if success else 'failed'}")
