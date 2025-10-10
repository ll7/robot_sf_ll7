FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
RUN python -m pip install uv
WORKDIR /app
ADD ./uv.toml ./uv.lock ./pyproject.toml ./
ADD ./fast-pysf ./fast-pysf
ADD ./robot_sf ./robot_sf
RUN uv sync --all-extras --frozen
ADD ./tests ./tests

WORKDIR /app/tests
# RUN ln -s ../fast-pysf/tests pysf_tests
WORKDIR /app
RUN uv run pytest tests
