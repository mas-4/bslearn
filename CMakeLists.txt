cmake_minimum_required(VERSION 3.24)
project(bslearn-tests C)

set(CMAKE_C_STANDARD 17)

if (CMAKE_BUILD_TYPE EQUAL "RELEASE")
    if (WIN32)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O2 /fp:fast")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -ffast-math")
    endif()
endif()

add_subdirectory(src)

set(TEST_SRCS
        tests/main.c)

add_executable(test ${TEST_SRCS})
target_include_directories(test PUBLIC include)

target_link_libraries(test bluesky_learn)
