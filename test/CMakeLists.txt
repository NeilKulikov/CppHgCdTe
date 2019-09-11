project(tests)

find_package (Boost COMPONENTS unit_test_framework REQUIRED)
find_package (GSL REQUIRED)

set(test1_source integration.cpp)
set(test2_source splines.cpp)
set(test3_source model.cpp)
set(test4_source matrix.cpp)

add_executable(test1 ${test1_source})
add_executable(test2 ${test2_source})
add_executable(test3 ${test3_source})
add_executable(test4 ${test4_source})

target_link_libraries(test1 ${Boost_LIBRARIES} GSL::gsl)
target_link_libraries(test2 ${Boost_LIBRARIES} GSL::gsl)
target_link_libraries(test3 ${Boost_LIBRARIES} GSL::gsl)
target_link_libraries(test4 ${Boost_LIBRARIES} GSL::gsl)

add_test(NAME Integration COMMAND test1)
add_test(NAME Splines COMMAND test2)
add_test(NAME Models COMMAND test3)
add_test(NAME Matrix COMMAND test4)