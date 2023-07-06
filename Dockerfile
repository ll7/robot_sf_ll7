FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
RUN python -m pip install pip --upgrade
WORKDIR /app/fast-pysf
ADD ./fast-pysf/requirements.txt .
RUN python -m pip install -r requirements.txt
ADD ./fast-pysf .
RUN python -m pip install .
WORKDIR /app
ADD ./requirements.txt .
RUN python -m pip install -r requirements.txt
ADD ./setup.py ./setup.py
ADD ./robot_sf ./robot_sf
RUN python -m pip install .
ADD ./tests ./tests

WORKDIR /app/tests
RUN ln -s ../fast-pysf/tests pysf_tests
WORKDIR /app
RUN python -m pytest tests
