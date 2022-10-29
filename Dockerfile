FROM tensorflow/tensorflow:latest-gpu
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
ADD ./simulation_zero_load.py ./simulation_zero_load.py
ENTRYPOINT ["python", "-m", "scalene", "simulation_zero_load.py"]
