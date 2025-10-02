"""Setup script to create a Python package for the project."""

from setuptools import find_packages, setup

setup(
    name="depth_anything_v2_qat",
    version="1.0.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
)
