# robot-sf

## About
Environment for the Simulation of a Robot moving
in a pedestrian-filled space.

## Quickstart

1) Clone Source Code

```sh
git clone https://git.rz.uni-augsburg.de/troestma/scoomatic-pysocialforce
cd robot-sf
```

2) Install Dependencies

```sh
python3 -m pip install pip --upgrade
python3 -m pip install -r requirements.txt
```

3) Run Linter / Tests / Code Analysis

3.1) Pylint / Pytest

```sh
python3 -m pytest
python3 -m pylint robot_sf --fail-under=9.5
```

3.2) SonarLint / SonarQube with VSCode

```sh
sudo snap install code --classic
code --install-extension sonarsource.sonarlint-vscode
```

```sh
docker-compose -f sonarqube-compose.yml up -d
```

```sh
sonar-scanner \
  -Dsonar.projectKey=scoomatic-pysocialforce \
  -Dsonar.python.version=3.8 \
  -Dsonar.sources=. \
  -Dsonar.host.url=http://localhost:9000 \
  -Dsonar.login=<your-sonarqube-token>
```

4) Run Simulation Benchmark

```sh
python3 simulation_benchmark_zero_load.py
```

## Original Repository
This repository is a fork of https://github.com/EnricoReg/robot-sf.
Thank you for providing the initial inspiration and technical
implementation to the project. This really helped me to achieve my goals.
