cmake_minimum_required(VERSION 3.10)
project(clcs)

# Set option for serialization support
set(CR_CLCS_BUILD_S11N OFF CACHE BOOL "Enable serialization support" FORCE)

# Source files
set(CR_CLCS_SRC_FILES
        src/segment.cc
        src/curvilinear_coordinate_system.cc
        src/projection_domain.cc
        src/path_segments.cc
        src/util.cc
        src/util_sweep_line.cc
        src/util_projection_domain.cc
)

# Header files
set(CR_CLCS_HDR_FILES
        include/geometry/application_settings.h
        include/geometry/clcs_exceptions.h
        include/geometry/clcs_types.h
        include/geometry/clcs_logger.h
        include/geometry/segment.h
        include/geometry/curvilinear_coordinate_system.h
        include/geometry/projection_domain.h
        include/geometry/path_segments.h
        include/geometry/util.h
        include/geometry/util_sweep_line.h
        include/geometry/util_projection_domain.h
        include/geometry/serialize/basic_types.h
        include/geometry/serialize/curvilinear_coordinate_system_export.h
        include/geometry/serialize/icurvilinear_coordinate_system_export.h
        include/geometry/serialize/serialize.h
        include/geometry/serialize/serialize_reg_impl.h
        include/geometry/serialize/vector2d_export.h
        include/geometry/serialize/vector2d_export_streams.h
        include/geometry/serialize/export_structs/curvilinear_coordinate_system_export_struct.h
        include/geometry/serialize/public/serialize_public.h
)

# Create library
add_library(crccosy ${CR_CLCS_SRC_FILES})
add_library(clcs ALIAS crccosy)
target_sources(crccosy PRIVATE ${CR_CLCS_HDR_FILES})

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.23.0)
    target_sources(crccosy
            INTERFACE
            FILE_SET crccosy_headers
            TYPE HEADERS
            BASE_DIRS include
            FILES ${CR_CLCS_HDR_FILES})
endif ()

# Configuration
target_compile_features(crccosy PUBLIC cxx_lambdas cxx_auto_type)
target_include_directories(crccosy
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Dependencies
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(crccosy PRIVATE OpenMP::OpenMP_CXX)
endif()

if(TARGET Eigen3::Eigen)
    target_link_libraries(crccosy PUBLIC Eigen3::Eigen)
endif()

# Link fmt
find_package(fmt REQUIRED)
if(TARGET fmt::fmt)
    target_link_libraries(crccosy PUBLIC fmt::fmt)
    message(STATUS "CLCS: fmt linked")
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FMT REQUIRED fmt)
    target_link_libraries(crccosy PUBLIC ${FMT_LIBRARIES})
    target_include_directories(crccosy PUBLIC ${FMT_INCLUDE_DIRS})
    target_compile_options(crccosy PUBLIC ${FMT_CFLAGS_OTHER})
endif()

# Serialization support
if (CR_CLCS_BUILD_S11N AND TARGET s11n)
    message(STATUS "CLCS: Serialization enabled")
    target_link_libraries(crccosy PUBLIC s11n)
    target_compile_definitions(crccosy PRIVATE -DENABLE_SERIALIZER=1)
else ()
    message(STATUS "CLCS: Serialization disabled")
    target_compile_definitions(crccosy PRIVATE -DENABLE_SERIALIZER=0)
endif ()

add_library(CommonRoadCLCS::crccosy ALIAS crccosy)

# Compiler warnings for top-level builds
if (${PROJECT_NAME}_IS_TOP_LEVEL)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(crccosy PRIVATE -Wall -Wextra -Wconversion
                -pedantic -Wfatal-errors -Wno-unused-parameter -Wno-unused-function -Wno-unused-but-set-variable)
    endif ()
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(crccosy PRIVATE /W3)
    endif ()
endif ()
