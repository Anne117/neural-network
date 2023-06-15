cmake_minimum_required(VERSION 3.5)

project(NetWorkForVideoOne)

set(CMAKE_CXX_STANDARD 17)

add_executable(nn NetWork.cpp NetWork.h Matrix.cpp Matrix.h ActivateFunction.cpp ActivateFunction.h source.cpp)

enable_testing()

add_executable(tests NetWork.cpp NetWork.h Matrix.cpp Matrix.h ActivateFunction.cpp ActivateFunction.h unittests.cpp)