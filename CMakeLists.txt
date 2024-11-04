cmake_minimum_required(VERSION 2.6)
project(ssl_fe)


find_package(Boost REQUIRED COMPONENTS timer thread)
find_package(PCL 1.8 REQUIRED)
find_package(Eigen3)
find_package(Ceres REQUIRED)
# find_package(GTSAM REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
include_directories(include())
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

link_directories(
  include
  ${PCL_LIBRARY_DIRS}
  ${CERES_LIBRARY_DIRS}
  ${GTSAM_LIBRARY_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)


add_executable(calib src/ssl_fe.cpp)

target_link_libraries (calib ${EIGEN3_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBS})

target_link_libraries(calib
  PUBLIC
      OpenMP::OpenMP_CXX
 )


