# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 12:26:16 2021

@author: Matteo Caruso
"""

from setuptools import setup
from setuptools import find_packages

setup(name='robot_sf',
      version='1.0.0',
      description='This package allows implementing a "gym-style" environment for the mobile robot navigating the crowd',
      url='https://github.com/matteocaruso1993/robot_env',
      author='Matteo Caruso and Enrico Regolin',
      author_email='matteo.caruso@phd.units.it',
      license="GPLv3",
      #packages=['robot_env'],
      packages=find_packages(),
      package_data = {'robot_sf': ['utils/maps/*.json','utils/config/map_config.toml']},
      install_requires=['numpy','Pillow','matplotlib','pysocialforce','python-math','jsons','toml','natsort'],
      zip_safe=False,
      include_package_data=True,
      python_requires='>=3.6'
      )
