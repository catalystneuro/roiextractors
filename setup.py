from setuptools import setup, find_packages
import os

d = {}
exec(open("segmentationextractors/version.py").read(), None, d)
version = d['version']
pkg_name = "segmentationextractors"
here = os.path.dirname(__file__)
with open(here + r'\README.md') as rd:
    long_description = rd.open()

setup(
    name=pkg_name,
    version=version,
    author="Saksham Sharda, Ben Dichter",
    author_email="saksham20.sharda@gmail.com",
    description="Python module for extracting optical physiology ROIs and traces for various file types and formats",
    url="https://github.com/ben-dichter-consulting/segmentationextractors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={},
    install_requires=[
        'numpy>=1.18.1',
        'scipy>=0.13.0',
        'scikit-image>=0.9.3',
        'shapely>=1.2.14',
        'scikit-learn>=0.11',
        'pillow>=2.6.1',
        'future>=0.14',
        'pynwb>=1.2.0',
        'h5py>=2.10.0',
        'python-dateutil>=2.7.3',
        'sima @ git+https://github.com/losonczylab/sima@1.3.2#egg=sima'
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    )
)
