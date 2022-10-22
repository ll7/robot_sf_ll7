FROM tensorflow/tensorflow:latest-gpu
WORKDIR /app
ADD ./requirements.txt .
RUN python --version
RUN python -m pip install pip --upgrade
RUN python -m pip install -r requirements.txt
ADD . .
RUN python -m pip install .
ENTRYPOINT ["python", "-m", "scalene", "simulation_benchmark_zero_load.py"]
