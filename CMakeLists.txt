cmake_minimum_required(VERSION 2.6)
project(CdHgTe)

SET (CMAKE_LIBRARY_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/bin
        CACHE PATH
        "Single Directory for all"
    )

SET (CMAKE_RUNTIME_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/bin
        CACHE PATH
        "Single Directory for all"
    )

SET (CMAKE_ARCHIVE_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/lib
        CACHE PATH
        "Single Directory for all"
    )

set (CMAKE_CXX_STANDARD 17)

enable_testing()

find_package(Boost REQUIRED)

find_package(GSL REQUIRED)

include_directories(ext_deps)
include_directories(headers)

add_subdirectory( test build/test )