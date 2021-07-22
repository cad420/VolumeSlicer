

list(
        APPEND
        MPILargeVolumeVisApp_SRCS
        main.cpp
        MPILargeVolumeVisApplication.cpp
        LargeVolumeVisGUI.cpp
        ${PROJECT_SOURCE_DIR}/third_party/glad/src/glad.c
)
add_executable(MPILargeVolumeVisApp ${MPILargeVolumeVisApp_SRCS})
target_link_libraries(
        MPILargeVolumeVisApp
        PRIVATE
        VS_CORE
        SDL2-static
        imgui
        ${MPI_LIBRARIES}
        ${OPENGL_LIBRARIES}
)
target_include_directories(
        MPILargeVolumeVisApp
        PRIVATE
        ${PROJECT_SOURCE_DIR}/third_party
        ${MPI_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}/third_party/glad/include
)