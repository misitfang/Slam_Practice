cmake_minimum_required(VERSION 2.8)
project(3d)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Ceres
find_package(OpenCV 3 REQUIRED)
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)


find_package(Sophus REQUIRED)

# Eigen
include_directories(${OpenCV_INCLUDE_DIRS}
    "/usr/include/eigen3")
add_executable(reprojection3d2dmain src/reprojection3d2dmain.cpp)
add_executable(pointsmain src/pointsmain.cpp)#主程序可执行文件

add_library(test src/GetPoints.cpp src/reprojection_factor.cpp src/utility.cpp src/icp3d3d_factor.cpp)
target_link_libraries(pointsmain test ${OpenCV_LIBS})#可执行文件调用共享库中求解函数
target_link_libraries(reprojection3d2dmain test Sophus::Sophus ${OpenCV_LIBS})
# target_link_libraries(icp3d3d_factor test)
# target_link_libraries(GetPoints  ${OpenCV_LIBS})
target_link_libraries(test Sophus::Sophus ${OpenCV_LIBS})




# add_library(GetPoints SHARED src/GetPoints.cpp)#point3dEError共享库
# add_library(reprojection_factor SHARED src/reprojection_factor.cpp)
# add_library(utility SHARED src/utility.cpp)
# add_library(icp3d3d_factor SHARED src/icp3d3d_factor.cpp)

# target_link_libraries(pointsmain GetPoints icp3d3d_factor utility  ${OpenCV_LIBS})#可执行文件调用共享库中求解函数
# target_link_libraries(reprojection3d2dmain GetPoints reprojection_factor utility Sophus::Sophus ${OpenCV_LIBS})
# target_link_libraries(icp3d3d_factor utility)
# target_link_libraries(GetPoints  ${OpenCV_LIBS})
# target_link_libraries(reprojection_factor Sophus::Sophus ${OpenCV_LIBS})


