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

### 3. Run Linter / Tests / Code Analysis

#### 3.1. Pylint / Pytest

```sh
python3 -m pytest
python3 -m pylint robot_sf --fail-under=9.5
```

#### 3.2. SonarLint / SonarQube with VSCode

Initial VSCode Setup

```sh
sudo snap install code --classic
code --install-extension sonarsource.sonarlint-vscode
```

Launch Local SonarQube Service Instance

```sh
docker-compose -f sonarqube-compose.yml up -d
firefox http://localhost:9000
```

Run Analysis / Gather Feedback

```sh
sonar-scanner \
  -Dsonar.projectKey=scoomatic-pysocialforce \
  -Dsonar.python.version=3.8 \
  -Dsonar.sources=. \
  -Dsonar.host.url=http://localhost:9000 \
  -Dsonar.login=<your-sonarqube-token>
```

### 4. Run Simulation Benchmark

#### 4.1 Manual Deployment on Host Machine

```sh
# register packages with pip
pushd ./fast-pysf
  python3 -m pip install .
popd
python3 -m pip install .
```

```sh
# benchmark with scalene
python3 -m scalene benchmark.py
```

#### 4.2 Dockerized Deployment (GPU-Empowered)

```sh
CURRENT_TIMESTAMP="$(date --iso-8601)T$(date +%T)" docker-compose up --build
firefox ./profiles/profile.html
```

#### 4.3 Run Simulation with UI (for Debugging)

```sh
# render the game state with pygame and observe live
python3 debug_simulation.py
```

## Original Repository
This repository is a fork of https://github.com/EnricoReg/robot-sf.
Thank you for providing the initial inspiration and technical
implementation to the project. This really helped me to achieve my goals.
