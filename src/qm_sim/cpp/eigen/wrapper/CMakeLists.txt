cmake_minimum_required(VERSION 3.16.3)

find_package(Python3 REQUIRED)

pybind11_add_module(eigen_wrapper MODULE py_interface.cpp)
target_link_libraries(eigen_wrapper PUBLIC eigen)
install(TARGETS eigen_wrapper DESTINATION qm_sim/cpp/eigen)