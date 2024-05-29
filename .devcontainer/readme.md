# Devcontainer

Creating a Docker image that can create a Pygame window on Windows 11 involves several steps. Pygame is a popular set of Python modules designed for writing video games. It requires access to graphical libraries which can be a bit tricky to set up in Docker because Docker containers are typically run in headless environments (without a graphical interface). However, it can be done with the help of X server.

Here are the steps to create a Docker image for running a Pygame window on Windows 11:

1. **Install Docker**: Ensure that Docker is installed on your Windows 11 machine. You can download and install Docker Desktop from the official Docker website.

2. **Create Dockerfile**: Create a `Dockerfile` to specify the environment and dependencies.

3. **Configure X Server**: On Windows, you will need an X server to display the Pygame window. Xming or VcXsrv are common choices for Windows. Install and configure an X server to allow connections from your Docker container.

4. **Run Docker Container**: Start the Docker container and ensure it can connect to the X server on your Windows host.

Here is a step-by-step guide:

### Step 1: Install X Server on Windows

1. **Download and Install VcXsrv**:
   - Download VcXsrv from [sourceforge.net](https://sourceforge.net/projects/vcxsrv/).
   - Install VcXsrv using the default settings.
   - Run VcXsrv, selecting the "Multiple windows" option, and ensure the "Disable access control" option is checked.

### Step 2: Create the Dockerfile

Create a directory for your Docker setup and create a file named `Dockerfile` with the following content:

```Dockerfile
# Use the official Python base image
FROM python:3.9

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev \
    python3-opengl libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
    libsmpeg-dev libportmidi-dev libavformat-dev libswscale-dev libjpeg-dev libfreetype6-dev

# Install Pygame
RUN pip install pygame

# Set the display environment variable
ENV DISPLAY=host.docker.internal:0.0

# Copy the Pygame script into the container
COPY pygame_script.py /usr/src/app/

# Set the working directory
WORKDIR /usr/src/app

# Run the Pygame script
CMD ["python", "pygame_script.py"]
```

### Step 3: Create a Pygame Script

In the same directory, create a simple Pygame script named `pygame_script.py`:

```python
import pygame
import sys

# Initialize Pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption('Pygame Window')

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the screen with a color
    screen.fill((0, 128, 255))

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
```

### Step 4: Build and Run the Docker Container

1. **Build the Docker Image**:
   Open a terminal in the directory containing your `Dockerfile` and `pygame_script.py` and run:

   ```sh
   docker build -t pygame-app .
   ```

2. **Run the Docker Container**:
   Ensure your X server (VcXsrv) is running on Windows, then run the Docker container:

   ```sh
   docker run --rm -e DISPLAY=host.docker.internal:0.0 pygame-app
   ```

This command sets the `DISPLAY` environment variable to use the X server running on your Windows host. The `--rm` flag ensures the container is removed after it stops.

You should see the Pygame window appear on your Windows desktop, managed by the X server.

### Summary

By following these steps, you should be able to create a Docker image that can display a Pygame window on Windows 11 using an X server like VcXsrv. This setup allows you to leverage Docker's isolation and portability while still being able to run graphical applications.