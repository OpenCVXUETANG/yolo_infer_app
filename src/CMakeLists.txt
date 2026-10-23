cmake_minimum_required( VERSION 2.8 )

# 声明一个 cmake 工程
project(yolov8_tensorrt_demo)

# 设置编译模式
#set( CMAKE_BUILD_TYPE "Release" )

#添加OPENCV库
#指定OpenCV版本，代码如下
#find_package(OpenCV 4.5.4 REQUIRED)
#如果不需要指定OpenCV版本，代码如下
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
	./src/)


#添加OpenCV头文件
include_directories(${OpenCV_INCLUDE_DIRS})
# 添加CUDA11.4头文件
include_directories(/usr/local/cuda-11.4/include)
link_directories(/usr/local/cuda-11.4/lib64)
# tensorRT
include_directories(/usr/local/cuda/include)
link_directories(/usr/lib/arrch64-linux-gnu)

#显示OpenCV_INCLUDE_DIRS的值
message(${OpenCV_INCLUDE_DIRS})

FILE(GLOB_RECURSE TEST_SRC
	src/*.cpp
	)

# 添加一个可执行程序
# 语法：add_executable( 程序名 源代码文件 ）
add_executable(target main.cpp ${TEST_SRC})

# 将库文件链接到可执行程序上
target_link_libraries(target  ${LD_LIBRARY_PATH})
target_link_libraries(target  nvinfer)
target_link_libraries(target  cudart)
target_link_libraries(target  ${OpenCV_LIBS})
