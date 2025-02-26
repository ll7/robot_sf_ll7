# robot-sf

> 2024-03-28 Under Development. See <https://github.com/ll7/robot_sf_ll7/issues>.

## About

This project provides a training environment for the simulation of a robot moving
in a pedestrian-filled space.

The project interfaces with Faram Foundations "Gymnasium" (former OpenAI Gym)
to facilitate trainings with various
SOTA reinforcement learning algorithms like e.g. StableBaselines3.
For simulating the pedestrians, the SocialForce model is used via a dependency
on a fork of PySocialForce.

Following video outlines some training results where a robot with e-scooter
kinematics is driving at the campus of University of Augsburg using real
map data from OpenStreetMap.

![](./docs/video/demo_01.gif)

## Installation

This project currently only supports local installation. The [.devcontainer installation](./.devcontainer/readme.md) is deprecated.

### Local Installation

Install python >= 3.10 and <= 3.12. Python **3.12** is recommended.
The following assumes that you are using the [uv python package manger](https://docs.astral.sh/uv/).

```sh
# Install pip
sudo apt-get update && sudo apt-get install -y python3-pip

# Install uv
pip install uv
```

### Clone Source Code

```sh
git clone --recurse-submodules https://github.com/Bonifatius94/robot-sf
cd robot-sf
```

### Create a `uv venv`

```sh
uv venv --python 3.12
source .venv/bin/activate
```

### Install Dependencies

```sh
uv pip install pip --upgrade
uv pip install -r requirements.txt
uv pip install -r fast-pysf/requirements.txt
```

### FFMPEG

For video recording of the simulation, ffmpeg is required.

```sh
sudo apt-get install -y ffmpeg
```

### Install your local packages in `editable` mode

```sh
uv pip install -e fast-pysf/. # pysocialforce
uv pip install -e . # robot_sf
```

### Install Pre-Commit Hook

```sh
pre-commit install
```

### Tests

#### Pysocialforce Tests (**currently not working**)

> [!WARNING]  
> Currently not working. See https://github.com/ll7/robot_sf_ll7/issues/1

Add symbolic link for pysocialforce and navigate to tests directory:

```sh
ln -s fast-pysf/pysocialforce pysocialforce
pushd tests
    # ln -s ../fast-pysf/tests pysf_tests
popd
```

*Note: The outlined command might differ on Windows, e.g. try mklink*

#### Run Linter / Tests

```sh
pytest tests
pylint robot_sf
```

#### GUI Tests

```sh
pytest test_pygame
```

### 5. Run Visual Debugging of Pre-Trained Demo Models

```sh
python3 examples/demo_offensive.py
python3 examples/demo_defensive.py
```

[Visualization](./docs/SIM_VIEW.md)

### 6. Run StableBaselines Training (Docker)

```sh
docker compose build && docker compose run \
    robotsf-cuda python ./scripts/training_ppo.py
```

*Note: See [this setup](./docs/GPU_SETUP.md) to install Docker with GPU support.*

> Older versions use `docker-compose` instead of `docker compose`.

### 7. Edit Maps

The preferred way to create maps: [SVG Editor](./docs/SVG_MAP_EDITOR.md)

### 8. Optimize Training Hyperparams (Docker)

```sh
docker-compose build && docker-compose run \
    robotsf-cuda python ./scripts/hparam_opt.py
```

### 9. Extension: Pedestrian as Adversarial-Agent

The pedestrian is an adversarial agent who tries to find weak points in the vehicle's policy.

The Environment is built according to gymnasium rules, so that multiple RL algorithms can be used to train the pedestrian.

It is important to know that the pedestrian always spawns near the robot.

![demo_ped](./docs/video/demo_ped.gif)

```sh
python3 examples/demo_pedestrian.py
```

[Visualization](./docs/SIM_VIEW.md)
