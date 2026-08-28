cmake_minimum_required(VERSION 3.10)

# Create frenetix library
set(FRENETIX_SOURCES
    CartesianSample.cpp
    CoordinateSystemWrapper.cpp
    CurvilinearSample.cpp
    TrajectorySample.cpp
    util.cpp
)

set(FRENETIX_HEADERS
    CartesianSample.hpp
    CoordinateSystemWrapper.hpp
    CurvilinearSample.hpp
    FillCoordinates.hpp
    TrajectorySample.hpp
    polynomial.hpp
    util.hpp
)

# Create the frenetix library
add_library(frenetix STATIC ${FRENETIX_SOURCES} ${FRENETIX_HEADERS})

# Set include directories for frenetix
target_include_directories(frenetix PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

# Add CLCS include directory if available
if(EXISTS ${CMAKE_SOURCE_DIR}/external/clcs/include)
    target_include_directories(frenetix PUBLIC ${CMAKE_SOURCE_DIR}/external/clcs/include)
endif()

# Link Eigen3 to frenetix
target_link_libraries(frenetix PUBLIC Eigen3::Eigen)

# Link CLCS library if available
if(TARGET crccosy)
    target_link_libraries(frenetix PUBLIC crccosy)
    message(STATUS "CLCS library linked to frenetix")
endif()

# Find and link spdlog
find_package(spdlog QUIET)
if(spdlog_FOUND)
    target_link_libraries(frenetix PUBLIC spdlog::spdlog)
    message(STATUS "spdlog found and linked to frenetix")
else()
    # Try to find spdlog as a system package
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(SPDLOG QUIET spdlog)
        if(SPDLOG_FOUND)
            target_link_libraries(frenetix PUBLIC ${SPDLOG_LIBRARIES})
            target_include_directories(frenetix PUBLIC ${SPDLOG_INCLUDE_DIRS})
            message(STATUS "spdlog found via pkg-config and linked to frenetix")
        else()
            message(WARNING "spdlog not found - frenetix may not compile properly")
        endif()
    else()
        message(WARNING "spdlog not found - frenetix may not compile properly")
    endif()
endif()

# Check if TaskFlow is available and link it if found
find_package(Taskflow QUIET)
if(Taskflow_FOUND)
    target_link_libraries(frenetix PUBLIC Taskflow::Taskflow)
    target_compile_definitions(frenetix PUBLIC FRENETIX_HAS_TASKFLOW)
    message(STATUS "TaskFlow found and linked to frenetix")
else()
    message(STATUS "TaskFlow not found - frenetix will be built without TaskFlow support")
endif()

# Set C++ standard
target_compile_features(frenetix PUBLIC cxx_std_17)

# Add compiler flags for optimization
target_compile_options(frenetix PRIVATE
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
    $<$<CONFIG:Debug>:-O0 -g>
)