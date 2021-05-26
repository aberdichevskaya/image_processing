project(work)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})



add_executable(work work.cpp "json.hpp")





