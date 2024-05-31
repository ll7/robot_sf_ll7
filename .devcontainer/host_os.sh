#!/bin/bash

# Set the HOST_OS environment variable based on the operating system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export HOST_OS=Linux
elif [[ "$OSTYPE" == "darwin"* ]]; then
    export HOST_OS=Mac
elif [[ "$OSTYPE" == "cygwin" ]]; then
    export HOST_OS=Windows
elif [[ "$OSTYPE" == "msys" ]]; then
    export HOST_OS=Windows
elif [[ "$OSTYPE" == "win32" ]]; then
    export HOST_OS=Windows
else
    export HOST_OS=Unknown
fi
echo $HOST_OS
