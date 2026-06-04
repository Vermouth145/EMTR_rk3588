cmake_minimum_required(VERSION 3.10)

project(rknn_yolo11_demo)


# 新增：查找OpenCV包
set(OpenCV_DIR "/home/forlinx/Documents/ffmpeg_rockchip_with_mpp/aarch64_rockchip/usr/local/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
link_directories("/home/forlinx/Documents/ffmpeg_rockchip_with_mpp/aarch64_rockchip/usr/local/lib")

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# 指定 FFmpeg 头文件路径
include_directories(/home/forlinx/Documents/ffmpeg_rockchip_with_mpp/aarch64_rockchip/usr/local/include)

# 指定 FFmpeg 库文件路径
link_directories(/home/forlinx/Documents/ffmpeg_rockchip_with_mpp/aarch64_rockchip/usr/local/lib)

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../../runtime/RK3588/${CMAKE_SYSTEM_NAME}/librknn_api)
set(LIB_ARCH aarch64)
set(RKNN_RT_LIB ${CMAKE_SOURCE_DIR}/include/librknnrt.so)
include_directories(${RKNN_API_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/3rdparty)

#rga
set(RGA_PATH ${CMAKE_SOURCE_DIR}/include/3rdparty/rga/RK3588)
set(RGA_LIB ${RGA_PATH}/lib/Linux/${LIB_ARCH}/librga.so)
include_directories( ${RGA_PATH}/include)
set(CMAKE_INSTALL_RPATH "lib")

# add multithread include path
include_directories(${CMAKE_SOURCE_DIR}/include)

#find_package (Eigen3 REQUIRED CONFIG QUIET)
set(Eigen3_INCLUDE_DIRS /home/forlinx/Documents/qh/v2/my_rknn_model/examples/yolo11/cpp/include/eigen3)

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

#set(rknpu_yolo11_file rknpu2/yolo11.cc)


add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/ utils.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

add_executable(${PROJECT_NAME}
    main.cc
    postprocess_pre.cc
    preprocess2.cc
    lapjv.cpp
    resize_function.cc
    STrack.cpp
    kalmanFilter.cpp
    BYTETracker.cpp
    utils.cpp
    evaluation.cpp
    ffmpeg_encoder.cpp
    threat_analysis.cpp
)

add_executable(threat_analysis_demo
    threat_analysis_demo.cc
    threat_analysis.cpp
)

# 新增：添加OpenCV头文件路径
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${OpenCV_INCLUDE_DIRS}  # 新增
)

target_link_libraries(${PROJECT_NAME}
    imageutils
    fileutils
    imagedrawing    
    ${LIBRKNNRT}
    ${OpenCV_LIBS}  # 新增
    ${RKNN_RT_LIB}
    ${RGA_LIB}
    ${Eigen3_LIBRARY}
    dl
    # 显式添加缺失的底层库
    jpeg
    png
    webp
    z
    m
    dl
    pthread
    swscale
    swresample
    avcodec
    avformat
    avutil
    x264      # 必须手动添加
)

#链接phtead库
find_package(Threads REQUIRED)
target_link_libraries(rknn_yolo11_demo ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(rknn_yolo11_demo ${OpenCV_LIBS})

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(${PROJECT_NAME}
    log
)
endif()

message(STATUS "!!!!!!!!!!!CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} Threads::Threads)
endif()

# Currently zero copy only supports rknpu2, v1103/rv1103b/rv1106 supports zero copy by default
install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/EMTR_labels_list.txt DESTINATION model)
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)
