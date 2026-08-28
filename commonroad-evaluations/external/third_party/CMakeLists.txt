cmake_minimum_required(VERSION 3.10)

# Add third-party libraries
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/libs11n/CMakeLists.txt)
    add_subdirectory(libs11n)
endif()

# Add LHS library
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/lhslib/CMakeLists.txt)
    add_subdirectory(lhslib)
endif()