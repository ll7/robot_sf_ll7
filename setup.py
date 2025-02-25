import os
from setuptools import setup, find_packages


PACKAGE_NAME = "robot_sf"
PACKAGE_VERSION = "2.0.0"
PACKAGE_AUTHORS = "Marco Tröster"
PACKAGE_DESCRIPTION = """This package allows implementing a "gymnasium-style" environment
for navigating a crowd with autonomous micromobility vehicles
"""
HOME_REPO = "https://github.com/Bonifatius94/robot_env"
EXCLUDE_FILES = []
PACKAGE_DATA = {"robot_sf": ["maps/*.json"]}
INSTALL_REQUIREMENTS = [
    "numpy",
    "gymnasium",
    "pylint",
    "pytest",
    "scalene",
    "numba",
    "pygame",
    "stable-baselines3",
    "tqdm",
    "rich",
    "tensorboard",
]
# TODO Update this package information


def get_ext_paths(root_dir, exclude_files):
    paths = []

    for root, _, files in os.walk(root_dir):
        for filename in files:
            if os.path.splitext(filename)[1] != ".py":
                continue
            if filename == "__init__.py":
                continue

            file_path = os.path.join(root, filename)
            if file_path in exclude_files:
                continue

            paths.append(file_path)

    return paths


setup(
    name=PACKAGE_NAME,
    version=PACKAGE_VERSION,
    description=PACKAGE_DESCRIPTION,
    url=HOME_REPO,
    author=PACKAGE_AUTHORS,
    author_email="marco.troester.student@uni-augsburg.de",
    license="GPLv3",
    packages=find_packages(),
    package_data=PACKAGE_DATA,
    install_requires=INSTALL_REQUIREMENTS,
    zip_safe=False,
    include_package_data=True,
    python_requires=">=3.10",
)
