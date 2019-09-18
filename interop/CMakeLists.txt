project(interop)

find_package (GSL REQUIRED)

set(lib_source hgcdte.cpp)

add_library(interop SHARED ${lib_source})

target_link_libraries(interop GSL::gsl)