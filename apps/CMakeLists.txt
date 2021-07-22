include(${PROJECT_SOURCE_DIR}/third_party/glfw.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/sdl2.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/imgui.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/msmpi.cmake)

find_package(OpenGL REQUIRED)

add_subdirectory(QTVolumeSlicer)

add_subdirectory(MPIVolumeSlicer)

add_subdirectory(MPILargeVolumeVis)