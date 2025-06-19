#!/usr/bin/env python3
"""Setup script for Renode Peripheral Generator CLI."""

from setuptools import setup, find_packages
import sys

if sys.version_info < (3, 8):
    sys.exit("Python 3.8 or higher is required")

with open("README_CLI.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="renode-peripheral-generator",
    version="1.0.0",
    author="James Drummond",
    author_email="james@example.com",
    description="CLI tool for generating Renode peripheral code using multi-agent AI systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/renode/renode-peripheral-generator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Code Generators",
        "Topic :: System :: Hardware",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "renode-generator=renode_generator:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json"],
    },
) 