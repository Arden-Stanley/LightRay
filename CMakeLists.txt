cmake_minimum_required(VERSION 4.1)

project(LightRay)

add_executable(lightray src/main.c src/window.c src/shader.c)

add_subdirectory(./libs/glfw-3.4)
target_link_libraries(lightray PUBLIC glfw)

add_subdirectory(./libs/glad)
target_link_libraries(lightray PUBLIC glad)
