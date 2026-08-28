# lex-stl-planner

## Installation

### System Dependencies

The following system packages are required to build the C++ bindings:

- **CMake** >= 3.20
- **Eigen3** >= 3.3
- **OpenMP**
- **yaml-cpp**
- **pybind11**
- **Pagmo2** (with Eigen3 support, required for the CMA-ES planner)
- **Boost** >= 1.68 (Pagmo2 dependency)
- **Intel TBB** (Pagmo2 dependency)

### Installing Pagmo2 (C++ library, from source)

Pagmo2 is a C++ library and **cannot** be installed via PDM/pip. It must be available as a system-level CMake package.

```bash
# Install dependencies
sudo apt-get install libboost-dev libboost-serialization-dev libtbb-dev libeigen3-dev

# Build and install pagmo2
git clone https://github.com/esa/pagmo2.git
cd pagmo2
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPAGMO_WITH_EIGEN3=ON
cmake --build . -j2
sudo cmake --build . --target install
```

The flag `-DPAGMO_WITH_EIGEN3=ON` is required — without it the `pagmo::cmaes` algorithm is not compiled.

### Python Environment

```bash
pdm install
```

### Building

```bash
./build.sh
```

## Usage

```bash
pdm run python run_cr_scenario.py
```

Available planner types: `"mppi"`, `"preemptive_mppi"`, `"random_shooting"`, `"const_input"`, `"frenet"`, `"cmaes"`, `"sa"`, `"de"`

## Reproducibility Experiments

Run these commands from the repository root:

```bash
pdm run python -m experiments.compare_simple_robustness
pdm run python -m experiments.compare_robustness_commonroad
pdm run python -m experiments.profile_robustness_evaluation
pdm run python -m experiments.compare_mppi_preemptive_mppi
pdm run python -m experiments.compare_planners
```
