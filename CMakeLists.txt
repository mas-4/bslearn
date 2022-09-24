cmake_minimum_required(VERSION 3.24)
project(bslearn-tests C)

set(CMAKE_C_STANDARD 17)

add_subdirectory(src)

set(TEST_SRCS
        tests/main.c)

add_executable(test ${TEST_SRCS})
target_include_directories(test PUBLIC include)

target_link_libraries(test bluesky_learn)
