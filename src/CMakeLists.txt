project(bluesky_learn C)

set(CMAKE_C_STANDARD 17)
if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif ()

set(C_SRCS
        layerdense.c
        common.c)
set(BS_INCLUDE
        ${CMAKE_SOURCE_DIR}/include)

add_library(bluesky_learn ${C_SRCS})

target_include_directories(bluesky_learn PRIVATE ${BS_INCLUDE})

# Configure MKL library, but not yet
set(BUILD_SHARED_LIBS OFF)

set(MKL_LINK static)
if (DEFINED ENV{MKLROOT})
    set(MKL_DIR $ENV{MKLROOT}/lib/cmake/mkl/)
    message(STATUS "MKLROOT is set to ${MKL_DIR}")
else ()
    set(MKL_DIR /opt/intel/oneapi/mkl/latest/lib/cmake/mkl/)
endif ()

find_package(MKL CONFIG)

if (MKL_FOUND)
    message(STATUS "MKL found")
    target_link_libraries(bluesky_learn PRIVATE MKL::MKL)
    target_include_directories(bluesky_learn PRIVATE ${MKL_H})
    configure_file(${OMP_LIBRARY} ${CMAKE_BINARY_DIR} COPYONLY)
    add_definitions(-DUSE_MKL)
else ()
    message(STATUS "MKL not found")
endif ()

if (NOT MKL_FOUND)
    find_package(BLAS)
    if (BLAS_FOUND)
        message(STATUS "BLAS found")
        target_link_libraries(bluesky_learn PRIVATE ${BLAS_LIBRARIES})
        target_include_directories(bluesky_learn PRIVATE ${BLAS_INCLUDE_DIRS})
        add_definitions(-DUSE_BLAS)
        if (APPLE)
            add_definitions(-DUSE_ACCELERATE)
        else ()
            add_definitions(-DUSE_OPENBLAS)
        endif ()
    else ()
        message(STATUS "BLAS not found")
    endif ()
endif ()

if (NOT WIN32)
    target_link_libraries(bluesky_learn PRIVATE m)
endif ()
