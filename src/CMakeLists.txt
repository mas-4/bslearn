project(bluesky_learn C)

set(CMAKE_C_STANDARD 17)

set(C_SRCS
        bslearn.c)
add_library(bluesky_learn ${C_SRCS})
target_include_directories(bluesky_learn PUBLIC ../include)
