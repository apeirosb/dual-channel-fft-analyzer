#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()
with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setup(
    name="dual-channel-fft-analyzer",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Antonio Peiro",
    author_email="",
    description="Python package that models and old Dual Channel FFT Analyzer, taking some of the functionality of the B&K Dual Channel Signal Analyzer Type 2032/2034 as a reference. The purpose of the application is merely educational to show concepts like autospectrum, cross-spectrum, coherence, frequency response estimation, etc.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests",)),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
)
