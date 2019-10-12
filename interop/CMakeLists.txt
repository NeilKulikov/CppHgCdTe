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
    set(libs_list ${Boost_LIBRARIES} GSL::gsl TBB::tbb)
else(PSTL)
    set(libs_list ${Boost_LIBRARIES} GSL::gsl)
endif(PSTL)

target_link_libraries(interop ${libs_list})