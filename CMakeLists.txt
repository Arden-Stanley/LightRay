cmake_minimum_required(VERSION 4.1)

project(LightRay)

add_executable(lightray src/main.cpp src/Shader.cpp src/Window.cpp src/Buffer.cpp)

target_compile_definitions(lightray PUBLIC SOURCE_DIR="${CMAKE_SOURCE_DIR}")

add_subdirectory(libs/glad)
add_subdirectory(libs/glfw-3.4)

target_link_libraries(lightray PUBLIC glad)
target_link_libraries(lightray PUBLIC glfw)