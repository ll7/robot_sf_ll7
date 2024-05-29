# Use the official Python base image
FROM python:3.9

# Set git to auto adjust for the correct line endings between different operating systems
RUN git config --global core.autocrlf true

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev \
    python3-opengl libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
    libsmpeg-dev libportmidi-dev libavformat-dev libswscale-dev libjpeg-dev libfreetype6-dev

# Install Pygame
RUN pip install pygame

# Set the display environment variable
ENV DISPLAY=host.docker.internal:0.0

# # Copy the Pygame script into the container
# COPY pygame_script.py /usr/src/app/

# # Set the working directory
# WORKDIR /usr/src/app

# # Run the Pygame script
# CMD ["python", "pygame_script.py"]

# Update pip
RUN pip install --upgrade pip
