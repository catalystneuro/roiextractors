from setuptools import setup, find_packages

d = {}
exec(open("roiextractors/version.py").read(), None, d)
version = d['version']
pkg_name = "roiextractors"
with open('README.md') as rd:
    long_description = rd.read()

setup(
    name=pkg_name,
    version=version,
    author="Saksham Sharda, Ben Dichter, Alessio Buccino",
    author_email="saksham20.sharda@gmail.com",
    description="Python module for extracting optical physiology ROIs and traces for various file types and formats",
    url="https://github.com/catalystneuro/roiextractors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={},
    install_requires=[
        'spikeextractors',
        'tqdm',
        'lazy_ops',
        'dill',
        'scipy',
        'PyYAML'
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research"
    )
)
