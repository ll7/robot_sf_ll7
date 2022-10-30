# robot-sf

## About
Environment for the Simulation of a Robot moving
in a pedestrian-filled space.

## Quickstart

### 1. Clone Source Code

```sh
git clone --recurse-submodules https://git.rz.uni-augsburg.de/troestma/scoomatic-pysocialforce
cd robot-sf
```

### 2. Install Dependencies

```sh
python3 -m pip install pip --upgrade
python3 -m pip install -r requirements.txt
python3 -m pip install -r fast-pysf/requirements.txt
```

```sh
pushd ./fast-pysf
  python3 -m pip install .
popd
python3 -m pip install .
```

*Note: 'python3 -m pip install .' needs to be executed to propagate robot_sf changes.*

### 3. Run Linter / Tests

```sh
python3 -m pytest
python3 -m pylint robot_sf --fail-under=9.5
```

### 4. Run OpenAI StableBaselines Training

```sh
docker-compose run robotsf-cuda python training.py
```
