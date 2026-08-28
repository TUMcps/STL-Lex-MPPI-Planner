bclib
=====

bertcarnell Template Library (bclib) for C++ utility headers

|<sub>Linux & MacOS</sub>|<sub>Windows</sub>|<sub>Code Coverage</sub>|<sub>Github Actions</sub>|
|:---:|:---:|:---:|:---:|
|[![Build Status](https://travis-ci.org/bertcarnell/bclib.svg?branch=master)](https://travis-ci.org/bertcarnell/bclib)|[![Build status](https://ci.appveyor.com/api/projects/status/ew4nolsitxr5sl5r?svg=true)](https://ci.appveyor.com/project/bertcarnell/bclib)|[![codecov](https://codecov.io/gh/bertcarnell/bclib/branch/master/graph/badge.svg)](https://codecov.io/gh/bertcarnell/bclib)|![CMake](https://github.com/bertcarnell/bclib/workflows/CMake/badge.svg)|

## Used in packages...

- [oa](https://github.com/bertcarnell/oa)
- [lhslib](https://github.com/bertcarnell/lhslib)
- [lhs](https://github.com/bertcarnell/lhs)

## Developing using CMake on Windows and Linux

### Ubuntu Linux

#### Prereqs

Install `cmake` for cross-platform make, and `lcov` for coverage statistics.

```
sudo apt-get install cmake lcov
```

#### CMake

```
cmake . -Bbuild -DCMAKE_BUILD_TYPE=Debug -DBUILD_TYPE=Coverage
cmake --build build
./build/bclibtest/bclibtest
cd build
lcov --directory bclibtest --capture --output-file bclibtest.info
lcov --remove bclibtest.info 'bclibtest/*' '/usr/*' --output-file bclibtest.clean.info
genhtml -o coverage bclibtest.clean.info
```

### Windows + Visual Studio

#### Prereqs

- Install [Visual Studio Community 2017](https://visualstudio.microsoft.com/vs/community/) with visual c++ support.
- Install [CMake](https://cmake.org/) for Windows

#### CMake

Open a Windows PowerShell window or Windows command prompt (cmd) to run these commands

```
cmake . -Bbuild -G "Visual Studio 15 2017 Win64"
```

#### Compile Option 1

Open Visual Studio.  File -> Open -> Project/Solution -> bcblib.sln

- right click on ALL_BUILD -> build
- right click on bcblibtest -> Debug -> Start New Instance

#### Compile Option 2

```
cmake --build build --target ALL_BUILD --config Release
./build/bclibtest/Release/bclibtest.exe
```
