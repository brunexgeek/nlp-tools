project(post)

cmake_minimum_required(VERSION 2.6)

set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Wl,--no-undefined -fPIC -std=gnu99 -Wall -Wconversion -Werror=implicit-function-declaration")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wl,--no-undefined -fPIC -Wall -Wconversion -Werror=implicit-function-declaration")

add_subdirectory("modules/post")
add_subdirectory("tools/train")
add_subdirectory("tools/predict")
add_subdirectory("tools/evaluate")
