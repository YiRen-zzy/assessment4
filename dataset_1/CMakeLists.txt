cmake_minimum_required(VERSION 3.10)

# 项目名称
project(ImageProcessor)

# 查找OpenCV库
find_package(OpenCV REQUIRED)

# 添加可执行文件
add_executable(process_images 2c.cpp)

# 链接OpenCV库
target_link_libraries(process_images ${OpenCV_LIBS})
