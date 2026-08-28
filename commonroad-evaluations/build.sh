#!/bin/bash
# Build script for lex-stl-planner with PDM environment

set -e

# Ensure PDM is available
export PATH="/home/patrick/.local/bin:$PATH"

# Create build directory if it doesn't exist
mkdir -p build
cd build

env -i \
    PATH="$PWD/../.venv/bin:/usr/bin:/bin:/usr/sbin" \
    HOME="/home/patrick" \
    PKG_CONFIG_PATH="/usr/lib/x86_64-linux-gnu/pkgconfig" \
    cmake \
        -DCMAKE_PREFIX_PATH="/usr/local;/usr;/usr/lib/x86_64-linux-gnu" \
        -DPYTHON_EXECUTABLE="$PWD/../.venv/bin/python" \
        -DCMAKE_BUILD_TYPE=Release \
        ..

env -i \
    PATH="$PWD/../.venv/bin:/usr/bin:/bin:/usr/sbin" \
    HOME="/home/patrick" \
    PKG_CONFIG_PATH="/usr/lib/x86_64-linux-gnu/pkgconfig" \
    make -j6