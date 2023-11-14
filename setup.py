#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name="center_surround",
    version="0.0",
    description="experiments to study center surround modulation",
    author="lucabaroni",
    packages=find_packages(exclude=[]),
    install_requires=[
        "deeplake",
    ],
)