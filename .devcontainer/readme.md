# Devcontainer

1. **Install Docker**: Ensure that Docker is installed on your machine. You can download and install Docker Desktop from the official Docker website.

2. **Configure X Server**: On Windows, you will need an X server to display the Pygame window. **VcXsrv** is a common choices for Windows. Install and configure an X server to allow connections from your Docker container.

3. **Start .devcontainer**: Open the project in Visual Studio Code and click on the green "Remote-Containers" icon in the bottom left corner. Select "Reopen in Container" to start the container.

Here is a step-by-step guide:

## X11 GUI support for the .devcontainer

> Unfortunately, the `remote-containers` extension does not support GUI applications out of the box. However, you can still run GUI applications in the container by connecting to an X server running on your host machine. We need to identify our host OS **manually** and follow the appropriate steps.

### Install X Server on Windows

1. **Download and Install VcXsrv**:
   - Download VcXsrv from [sourceforge.net](https://sourceforge.net/projects/vcxsrv/).
   - Install VcXsrv using the default settings.
   - Run VcXsrv, selecting the "Multiple windows" option, and ensure the "Disable access control" option is checked.

Inside the docker container, the `DISPLAY` environment variable should be set to `host.docker.internal:0.0` to connect to the X server running on your Windows host.

### X11 on ubuntu

enable x11 access for the docker container

```sh
# on the host system

# if docker requires sudo
sudo xhost +local:root

# if docker does not require sudo
xhost +local:docker
```

In the docker container, set the `DISPLAY` environment variable to:

```sh
# inside the container
export DISPLAY=:0
```

This command sets the `DISPLAY` environment variable to use the X server running on your host.

You should see the Pygame window appear on your Windows desktop, managed by the X server.
