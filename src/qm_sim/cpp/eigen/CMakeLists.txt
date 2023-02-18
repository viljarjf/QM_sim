cmake_minimum_required(VERSION 3.16.3)

find_package(Python3 REQUIRED)

# Sourcefiles for the library
add_library(eigen STATIC 
    src/HamiltonianBase.cpp
    src/eigensolver.cpp
    )

# Include directories for the library, also add external libraries
target_include_directories(eigen 
    PUBLIC include
    PUBLIC ${PROJECT_SOURCE_DIR}/lib/eigen
    PUBLIC ${PROJECT_SOURCE_DIR}/lib/spectra/include
    )

# Necessary for parallelization
set_property(TARGET eigen PROPERTY CXX_STANDARD 20)

add_custom_command( # Generate __init__.py file
    TARGET eigen
    COMMAND ${Python3_EXECUTABLE} scripts/generate_init.py eigen_wrapper
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
)

add_subdirectory(wrapper)