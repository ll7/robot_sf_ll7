# robot-sf

## About
This project provides a training environment for the simulation of a robot moving
in a pedestrian-filled space.

The project interfaces with OpenAI Gym to facilitate trainings with various
SOTA reinforcement learning algorithms like e.g. OpenAI StableBaselines3.
For simulating the pedestrians, the SocialForce model is used via a dependency
on a fork of PySocialForce.

## Quickstart

### 1. Clone Source Code

```sh
git clone --recurse-submodules https://github.com/Bonifatius94/robot-sf
cd robot-sf
```

### 2. Install Dependencies

```sh
python3 -m pip install pip --upgrade
python3 -m pip install -r requirements.txt
python3 -m pip install -r fast-pysf/requirements.txt
```

### 3. Register *robot_sf* and *pysocialforce* Packages 

```sh
pushd ./fast-pysf
    python3 -m pip install .
popd
python3 -m pip install .
```

*Note: This needs to be repeated to propagate robot_sf changes on a host deployment.
Otherwise the calling scope might run an old version of robot_sf.
Dockerized deployments will recognize when a change requires a rebuild.*

### 4. Run Linter / Tests

```sh
python3 -m pytest --ignore=fast-pysf
python3 -m pylint robot_sf --fail-under=9.5
```

### 5. Run StableBaselines Training (Docker)

```sh
docker-compose build && docker-compose run \
    robotsf-cuda python ./scripts/training.py
```

*Note: See [this setup](./GPU_SETUP.md) to install Docker with GPU support.*
