project(bluesky_learn C)

set(CMAKE_C_STANDARD 17)

set(C_SRCS
        bslearn.c
        layerdense.c
        common.c)
set(BS_INCLUDE
        ${CMAKE_SOURCE_DIR}/include)

set(BUILD_SHARED_LIBS OFF)

set(MKL_LINK static)
set(MKL_DIR /opt/intel/oneapi/mkl/latest/lib/cmake/mkl/)
find_package(MKL CONFIG REQUIRED)

get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()

add_library(bluesky_learn ${C_SRCS})

target_include_directories(bluesky_learn PRIVATE ${MKL_H})
target_include_directories(bluesky_learn PRIVATE ${BS_INCLUDE})

target_link_libraries(bluesky_learn PRIVATE MKL::MKL)

configure_file(${OMP_LIBRARY} ${CMAKE_BINARY_DIR} COPYONLY)