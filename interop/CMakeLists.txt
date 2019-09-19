project(interop)

find_package (GSL REQUIRED)

set(lib_source hgcdte.cpp)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(interop SHARED ${lib_source})

target_link_libraries(interop GSL::gsl)