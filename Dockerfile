FROM tensorflow/tensorflow:latest-gpu
WORKDIR /app
ADD ./requirements.txt .
RUN python --version
RUN python -m pip install pip --upgrade
RUN python -m pip install -r requirements.txt
ADD ./setup.py ./setup.py
ADD ./robot_sf ./robot_sf
RUN python -m pip install .
ADD ./benchmarks ./benchmarks
ENTRYPOINT ["python", "-m", "scalene", "./benchmarks/simulation_zero_load.py"]
