import json
import os
import tempfile
from pathlib import Path
from shutil import copy

# Output by default to a temporary directory
OUTPUT_PATH = Path(tempfile.mkdtemp())


# Load the configuration for the data tests


if os.getenv("CI"):
    LOCAL_PATH = Path(".")  # Must be set to "." for CI
    print("Running GIN tests on Github CI!")
else:
    # Override LOCAL_PATH in the `gin_test_config.json` file to a point on your system that contains the dataset folder
    # Use DANDIHub at hub.dandiarchive.org for open, free use of data found in the /shared/catalystneuro/ directory
    test_config_path = Path(__file__).parent / "gin_test_config.json"
    config_file_exists = test_config_path.exists()
    if not config_file_exists:

        root = test_config_path.parent.parent
        base_test_config_path = root / "base_gin_test_config.json"

        test_config_path.parent.mkdir(parents=True, exist_ok=True)
        copy(src=base_test_config_path, dst=test_config_path)

    with open(file=test_config_path) as f:
        # Load the configuration for the data tests
        test_config_dict = json.load(f)

    LOCAL_PATH = Path(test_config_dict["LOCAL_PATH"])

    if test_config_dict["SAVE_OUTPUTS"]:
        OUTPUT_PATH = LOCAL_PATH / "neuroconv_test_outputs"
        OUTPUT_PATH.mkdir(exist_ok=True, parents=True)


OPHYS_DATA_PATH = LOCAL_PATH / "ophys_testing_data"

TEXT_DATA_PATH = Path(__file__).parent.parent.parent / "tests" / "test_text"
