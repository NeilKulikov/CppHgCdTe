project(interop)

find_package(GSL REQUIRED)

message("Interop pstl: " ${PSTL})

if(PSTL)
    find_package(TBB REQUIRED)
    add_definitions(-DUSE_PSTL)
endif(PSTL)

set(lib_source hgcdte.cpp)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(interop SHARED ${lib_source})

if(PSTL)
    target_link_libraries(interop GSL::gsl TBB::tbb)
else(PSTL)
    target_link_libraries(interop GSL::gsl)
endif(PSTL)