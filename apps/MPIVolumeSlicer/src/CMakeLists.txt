
include(${PROJECT_SOURCE_DIR}/third_party/glfw.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/sdl2.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/imgui.cmake)
list(
        APPEND
        MPIVolumeSliceApp_SRCS
        main.cpp
        MPIVolumeSliceApplication.cpp
        VolumeSliceGUI.cpp
        ${PROJECT_SOURCE_DIR}/third_party/glad/src/glad.c
)
add_executable(MPIVolumeSliceApp ${MPIVolumeSliceApp_SRCS})
target_link_libraries(
        MPIVolumeSliceApp
        PRIVATE
        VS_CORE
        SDL2-static
        imgui
)
target_include_directories(
        MPIVolumeSliceApp
        PRIVATE
        ${PROJECT_SOURCE_DIR}/third_party
)