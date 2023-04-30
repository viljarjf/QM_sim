cmake_minimum_required(VERSION 3.16.3)

project(qm_sim)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

# Commented out to remove need to build the cpp stuff.
# The C++ files will stay for now, in case they are expanded upon in the future
# add_subdirectory(src/qm_sim/cpp)
