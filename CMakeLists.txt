cmake_minimum_required(VERSION 3.5.1)
project(mobilenet)


SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/)

# Try to find OpenCV
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    # If the package has been found, several variables will
    # be set, you can find the full list with descriptions
    # in the OpenCVConfig.cmake file.
    # Print some message showing some of them
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "Could not locate OpenCV")
endif()

# Try to find Boost
find_package(Boost COMPONENTS program_options filesystem REQUIRED)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
else ()
    message(FATAL_ERROR "Could not locate Boost")
endif()

file(GLOB SOURCE_FILES src/*.cpp)

#include_directories(${PROJECT_SOURCE_DIR}/include/ ${PROJECT_SOURCE_DIR}/ncnn-20210322-ubuntu-1804-shared/include/ ~/vulkan/1.2.170.0/x86_64/include/)
#link_directories(${PROJECT_SOURCE_DIR}/ncnn-20210322-ubuntu-1804-shared/lib/)

include_directories(${PROJECT_SOURCE_DIR}/include/ /media/lihongjie/Windows/work/code/ncnn/build-host-gcc-linux/install/include/ ~/vulkan/1.2.170.0/x86_64/include/)
link_directories(/media/lihongjie/Windows/work/code/ncnn/build-host-gcc-linux/install/lib/)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries (
        ${CMAKE_PROJECT_NAME}
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        pthread
        ncnn
        vulkan
)
