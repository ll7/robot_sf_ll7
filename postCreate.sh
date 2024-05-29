#!/bin/bash

git submodule update --init --recursive
pip install -r requirements.txt
pip install -r fast-pysf/requirements.txt
pip install -e .
pip install -e fast-pysf
