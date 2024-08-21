cmake_minimum_required(VERSION 3.19)
project(OS3)

set(CMAKE_C_STANDARD 11)

include_directories(.)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

add_executable(OS3 main.cpp MapReduceFramework.h MapReduceFramework.cpp MapReduceClient.h test3.cpp)