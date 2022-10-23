import os
from setuptools import setup, find_packages
from Cython.Build import cythonize


PACKAGE_NAME = 'robot_sf'
PACKAGE_VERSION = '1.0.0'
PACKAGE_AUTHORS = 'Matteo Caruso and Enrico Regolin'
PACKAGE_DESCRIPTION = """
This package allows implementing a "gym-style" environment
for the mobile robot navigating the crowd
"""
HOME_REPO = 'https://github.com/matteocaruso1993/robot_env'
EXCLUDE_FILES = []
PACKAGE_DATA = {'robot_sf': ['utils/maps/*.json','utils/config/map_config.toml']}
INSTALL_REQUIREMENTS = ['numpy', 'Pillow', 'matplotlib', 'pysocialforce', 'python-math',
                        'jsons', 'toml', 'natsort', 'numba', 'shapely']


def get_ext_paths(root_dir, exclude_files):
    paths = []

    for root, _, files in os.walk(root_dir):
        for filename in files:
            if os.path.splitext(filename)[1] != '.py':
                continue
            if filename=='__init__.py':
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
    author_email='matteo.caruso@phd.units.it',
    license="GPLv3",
    packages=find_packages(),
    package_data=PACKAGE_DATA,
    install_requires=INSTALL_REQUIREMENTS,
    zip_safe=False,
    # ext_modules = cythonize(
    #     get_ext_paths('robot_sf', EXCLUDE_FILES),
    #     compiler_directives={'language_level':3}),
    include_package_data=True,
    python_requires='>=3.8'
)
