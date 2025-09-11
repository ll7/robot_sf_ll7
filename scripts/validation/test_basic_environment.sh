#!/bin/bash
# Basic Environment Creation Test
# Tests core simulation setup functionality

set -e  # Exit on any error

echo "Testing basic environment creation..."

export DISPLAY=
export MPLBACKEND=Agg
export SDL_VIDEODRIVER=dummy

uv run python -c "
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
from robot_sf.gym_env.environment_factory import make_robot_env
print('Testing environment creation...')
env = make_robot_env(debug=True)
print('Environment created successfully')
obs, _ = env.reset()
print('Environment reset successful')
print('Observation space:', env.observation_space)
print('Action space:', env.action_space)
env.close()
print('Basic validation completed successfully')
"

echo "✅ Basic environment creation test passed"