cmake_minimum_required(VERSION 3.16.3)

project(qm_sim)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(src/qm_sim/cpp)
