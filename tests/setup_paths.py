import os
import json
import tempfile
from pathlib import Path

import pytest


# Load the configuration for the data tests
file_path = Path(__file__).parent.parent.parent / "tests" / "test_on_data" / "gin_test_config.json"

with open(file=file_path) as f:
    test_config_dict = json.loads(f)

if os.getenv("CI"):
    LOCAL_PATH = Path(".")  # Must be set to "." for CI
    print("Running GIN tests on Github CI!")
else:
    # Override LOCAL_PATH in the `gin_test_config.json` file to a point on your system that contains the dataset folder
    # Use DANDIHub at hub.dandiarchive.org for open, free use of data found in the /shared/catalystneuro/ directory
    LOCAL_PATH = Path(test_config_dict["LOCAL_PATH"])
    print("Running GIN tests locally!")

OPHYS_DATA_PATH = LOCAL_PATH / "ophys_testing_data"
if not OPHYS_DATA_PATH.exists():
    pytest.fail(f"No folder found in location: {OPHYS_DATA_PATH}!")

if test_config_dict["SAVE_OUTPUTS"]:
    OUTPUT_PATH = LOCAL_PATH / "example_nwb_output"
    OUTPUT_PATH.mkdir(exist_ok=True)
else:
    OUTPUT_PATH = Path(tempfile.mkdtemp())
