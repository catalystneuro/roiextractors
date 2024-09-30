from roiextractors.tools.plotting import show_video
from roiextractors.tools.testing import generate_mock_imaging_extractor


def test_show_video():
    imaging_extractor = generate_mock_imaging_extractor()
    anim = show_video(imaging=imaging_extractor)
