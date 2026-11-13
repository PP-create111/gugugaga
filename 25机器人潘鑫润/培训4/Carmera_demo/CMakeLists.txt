cmake_minimum_required(VERSION 3.10)
project(Carmera_demo)

find_package(OpenCV REQUIRED)

add_executable(main main.cpp)

target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main PRIVATE ${OpenCV_LIBS})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cam_mix_img DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
