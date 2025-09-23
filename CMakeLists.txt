cmake_minimum_required(VERSION 3.3)

project(LightRay)

add_executable(lightray src/main.cpp src/Window.cpp src/Shader.cpp src/Buffer.cpp)

add_subdirectory(./libs/glfw-3.4)
target_link_libraries(lightray PUBLIC glfw)

add_subdirectory(./libs/glad)
target_link_libraries(lightray PUBLIC glad)
