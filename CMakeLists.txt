cmake_minimum_required(VERSION 3.15...3.26)

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(dependencies)

find_package(pybind11 CONFIG REQUIRED)
find_package(Armadillo CONFIG REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)
find_package(GSL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Kokkos REQUIRED)

file(GLOB PGAS_SRCS CONFIGURE_DEPENDS src/spike_find/pgas/*.cpp)

pybind11_add_module(pgas_bound MODULE src/spike_find/pgas/bindings.cpp ${PGAS_SRCS} NO_EXTRAS)

target_include_directories(pgas_bound PRIVATE src/spike_find/pgas)

target_link_libraries(pgas_bound PRIVATE armadillo JsonCpp::JsonCpp GSL::gsl GSL::gslcblas)
target_link_libraries(pgas_bound PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(pgas_bound PRIVATE Kokkos::kokkos)

install(TARGETS pgas_bound DESTINATION ${SKBUILD_PROJECT_NAME}/pgas)
