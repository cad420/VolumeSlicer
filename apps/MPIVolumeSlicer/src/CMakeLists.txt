

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
        ${MPI_CXX_LIBRARIES}
        ${OPENGL_LIBRARIES}
)
target_include_directories(
        MPIVolumeSliceApp
        PRIVATE
        ${PROJECT_SOURCE_DIR}/third_party
        ${MPI_CXX_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/third_party/glad/include
)