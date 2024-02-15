from pathlib import Path
from setuptools import setup, find_packages
from shutil import copy as copy_file


root = Path(__file__).parent
with open(root / "README.md") as f:
    long_description = f.read()
with open(root / "requirements-minimal.txt") as f:
    install_requires = f.readlines()
with open(root / "requirements-full.txt") as f:
    full_dependencies = f.readlines()
with open(root / "requirements-testing.txt") as f:
    testing_dependencies = f.readlines()
extras_require = dict(full=full_dependencies, test=testing_dependencies)

# Create a local copy for the gin test configuration file based on the master file `base_gin_test_config.json`
gin_config_file_base = root / "base_gin_test_config.json"
gin_config_file_local = root / "tests" / "gin_test_config.json"
if not gin_config_file_local.exists():
    copy_file(src=gin_config_file_base, dst=gin_config_file_local)

setup(
    name="roiextractors",
    version="0.5.6",
    author="Heberto Mayorquin, Szonja Weigl, Cody Baker, Ben Dichter, Alessio Buccino",
    author_email="ben.dichter@gmail.com",
    description="Python module for extracting optical physiology ROIs and traces for various file types and formats",
    url="https://github.com/catalystneuro/roiextractors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,  # Includes files described in MANIFEST.in in the installation.
    python_requires=">=3.7",
    install_requires=install_requires,
    extras_require=extras_require,
)
