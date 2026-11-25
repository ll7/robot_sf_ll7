FROM python:3.12.3-slim-bookworm

ARG UV_VERSION=0.13.4

ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY= \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1 \
    PYGAME_HIDE_SUPPORT_PROMPT=1 \
    PYTHONUNBUFFERED=1 \
    SDL_VIDEODRIVER=dummy \
    TF_CPP_MIN_LOG_LEVEL=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        git \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-deps "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70" \
    && python -m pip install "uv==${UV_VERSION}"

WORKDIR /workspace/robot_sf_ll7

COPY pyproject.toml uv.lock ./
COPY fast-pysf ./fast-pysf
COPY third_party/python-rvo2 ./third_party/python-rvo2
RUN uv sync --all-extras --frozen --no-install-project

COPY . .
RUN rm -rf third_party/python-rvo2/build
RUN uv sync --all-extras --frozen
ENV NODE_OPTIONS=--use-openssl-ca
RUN .venv/bin/python scripts/ci/check_tls_trust.py

ENTRYPOINT ["scripts/repro/benchmark_bundle_smoke.sh"]
