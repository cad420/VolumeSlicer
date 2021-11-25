include(${PROJECT_SOURCE_DIR}/third_party/glfw.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/sdl2.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/imgui.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/msmpi.cmake)

find_package(OpenGL REQUIRED)

set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/OffScreenVolumeRenderer/third_party/opencv)
find_package(OpenCV REQUIRED)

add_subdirectory(QTVolumeSlicer)

add_subdirectory(MPIVolumeSlicer)

add_subdirectory(MPILargeVolumeVis)

add_subdirectory(OffScreenVolumeRenderer)

add_subdirectory(WebSlicerServer)

add_subdirectory(MPILargeMeshVis)

add_subdirectory(QTOffScreenRenderEditor)