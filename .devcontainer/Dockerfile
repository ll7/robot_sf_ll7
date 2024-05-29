# Use the official Python base image
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-utils 2>&1

# Verify git and needed tools are installed
RUN apt-get install --no-install-recommends -y git

# Install necessary dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    python3 python-is-python3 python3-pip python3-dev python3.10-venv \
    python3-opengl libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
    libsmpeg-dev libportmidi-dev libavformat-dev libswscale-dev libjpeg-dev libfreetype6-dev

# Set git to auto adjust for the correct line endings between different operating systems
RUN git config --global core.autocrlf true

# Set the display environment variable
ENV DISPLAY=host.docker.internal:0.0

# Create a virtual environment and activate it
# Avoids pip complaint about being a root user in the docker container
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Update pip
RUN pip install --upgrade pip
