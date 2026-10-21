cmake_minimum_required(VERSION 3.16.3)

project(sp_vision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
message(STATUS "--------------------CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}--------------------")

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake/")
find_package(OpenVINO REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

set(SENTRY_SPECIAL_MODE "OFF" CACHE STRING "Build sentry-specific ROS2 gimbal and ovsentry targets: AUTO, ON, or OFF")
set_property(CACHE SENTRY_SPECIAL_MODE PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${SENTRY_SPECIAL_MODE}" SENTRY_SPECIAL_MODE)
set(SENTRY_SPECIAL_MODE "${SENTRY_SPECIAL_MODE}" CACHE STRING "Build sentry-specific ROS2 gimbal and ovsentry targets: AUTO, ON, or OFF" FORCE)

set(_sentry_special_mode_values AUTO ON OFF)
if(NOT SENTRY_SPECIAL_MODE IN_LIST _sentry_special_mode_values)
    message(FATAL_ERROR "Invalid SENTRY_SPECIAL_MODE='${SENTRY_SPECIAL_MODE}'. Expected one of: AUTO, ON, OFF.")
endif()

add_subdirectory(tools)
add_subdirectory(io)
add_subdirectory(tasks/auto_aim)
add_subdirectory(tasks/auto_buff)
add_subdirectory(tasks/omniperception)

####################src###################
file(GLOB OVGIMBAL_SOURCES CONFIGURE_DEPENDS src/ovgimbal*.cpp)
foreach(OVGIMBAL_SOURCE ${OVGIMBAL_SOURCES})
    get_filename_component(OVGIMBAL_TARGET ${OVGIMBAL_SOURCE} NAME_WE)
    add_executable(${OVGIMBAL_TARGET} ${OVGIMBAL_SOURCE})
    target_link_libraries(${OVGIMBAL_TARGET} ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io)
endforeach()

##################calibration################
file(GLOB CALIBRATION_SOURCES CONFIGURE_DEPENDS calibration/*.cpp)
foreach(CALIBRATION_SOURCE ${CALIBRATION_SOURCES})
    get_filename_component(CALIBRATION_TARGET ${CALIBRATION_SOURCE} NAME_WE)
    add_executable(${CALIBRATION_TARGET} ${CALIBRATION_SOURCE})
    target_link_libraries(${CALIBRATION_TARGET} ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
endforeach()

##################tests##################
add_executable(auto_aim_test tests/auto_aim_test.cpp)
add_executable(auto_buff_test tests/auto_buff_test.cpp)

target_link_libraries(auto_aim_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)
target_link_libraries(auto_buff_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_buff tools io)

if(TARGET io_ros2_gimbal)
    find_package(fastcdr REQUIRED)
    add_executable(ovsentry_omni_mpc_debug src/ovsentry_omni_mpc_debug.cpp)
    target_link_libraries(ovsentry_omni_mpc_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff omniperception tools io io_ros2_gimbal fastcdr)
else()
    message(STATUS "io_ros2_gimbal target unavailable, skipping ovsentry_omni_mpc_debug.")
endif()
