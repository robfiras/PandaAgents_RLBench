cmake_minimum_required(VERSION 3.16)
project("sum_tree_cpp")
set(CMAKE_CXX_STANDARD 17)

set(MODULE_SOURCE "${PROJECT_SOURCE_DIR}/src/module")
set(EXTERNALS "${PROJECT_SOURCE_DIR}/externals")
set(PYBIND11_PYTHON_VERSION 3.7)

# pybind11
add_subdirectory(${EXTERNALS}/pybind11)

# module
pybind11_add_module (
    sum_tree_cpp
    ${MODULE_SOURCE}/module.cpp
    ${MODULE_SOURCE}/sum_tree.h
    ${MODULE_SOURCE}/sum_tree.cpp
    ${MODULE_SOURCE}/node.h
    ${MODULE_SOURCE}/node.cpp
)

target_include_directories (
    sum_tree_cpp
    PRIVATE ${MODULE_SOURCE}
)

