cmake_minimum_required(VERSION 3.15...3.26)

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(pgas MODULE src/spike_find/pgas/bindings.cpp)

install(TARGETS pgas DESTINATION .)
