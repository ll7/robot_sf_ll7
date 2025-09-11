#!/bin/bash
# Model Loading and Prediction Test
# Tests ML model integration functionality

set -e  # Exit on any error

echo "Testing model loading and prediction..."

export DISPLAY=
export MPLBACKEND=Agg
export SDL_VIDEODRIVER=dummy

uv run python -c "
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
from robot_sf.gym_env.environment_factory import make_robot_env
from stable_baselines3 import PPO
env = make_robot_env(debug=True)
# Use newest model for best compatibility
model = PPO.load('./model/ppo_model_retrained_10m_2025-02-01.zip', env=env)
obs, _ = env.reset()
action, _ = model.predict(obs, deterministic=True)
print('Model loading and prediction successful')
print('Action shape:', action.shape)
env.close()
"

echo "✅ Model loading and prediction test passed"