import os
from pathlib import Path
from setuptools import setup, find_packages
from copy import copy
from shutil import copy as copy_file

d = {}
exec(open("roiextractors/version.py").read(), None, d)
version = d["version"]
pkg_name = "roiextractors"


path = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(path, "README.md")) as f:
    long_description = f.read()
with open(os.path.join(path, "requirements-minimal.txt")) as f:
    install_requires = f.readlines()
with open(os.path.join(path, "requirements-full.txt")) as f:
    full_dependencies = f.readlines()
testing_dependencies = copy(full_dependencies)
with open(os.path.join(path, "requirements-testing.txt")) as f:
    testing_dependencies.extend(f.readlines())
extras_require = dict(full=full_dependencies, testing=testing_dependencies)

# Create a local copy for the gin test configuration file based on the master file `base_gin_test_config.json`
gin_config_file_base = Path("./base_gin_test_config.json")
gin_config_file_local = Path("./tests/gin_test_config.json")
if not gin_config_file_local.exists():
    copy_file(src=gin_config_file_base, dst=gin_config_file_local)

setup(
    name=pkg_name,
    version=version,
    author="Saksham Sharda, Cody Baker, Ben Dichter, Alessio Buccino",
    author_email="ben.dichter@gmail.com",
    description="Python module for extracting optical physiology ROIs and traces for various file types and formats",
    url="https://github.com/catalystneuro/roiextractors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={},
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-3 Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ),
)
