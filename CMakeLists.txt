cmake_minimum_required(VERSION 3.15)
project(classifier)

set(CMAKE_CXX_STANDARD 17)
find_package(Boost REQUIRED COMPONENTS filesystem)
include_directories(. ${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

add_executable(classifier classifier.cpp matplotlib.h util.h)
find_package(PythonLibs 3.6)
target_include_directories(classifier PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(classifier ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})