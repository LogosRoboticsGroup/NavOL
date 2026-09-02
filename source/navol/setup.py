# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'navol' python package."""

import os
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from setuptools import find_packages, setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
with open(os.path.join(EXTENSION_PATH, "config", "extension.toml"), "rb") as stream:
    EXTENSION_TOML_DATA = tomllib.load(stream)

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "casadi",
    "cvxpy",
    "imageio",
    "matplotlib",
    "numpy",
    "opencv-python",
    "open3d",
    "packaging",
    "psutil",
    "rpyc",
    "scipy",
    "toml",
    "tqdm",
    "trimesh",
]

# Installation operation
setup(
    name="navol",
    packages=find_packages(include=["navol", "navol.*"]),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="BSD-3-Clause",
    license_files=["LICENSE", "THIRD_PARTY_NOTICES.md"],
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: BSD License",
        "Isaac Sim :: 4.5.0",
    ],
    zip_safe=False,
)
