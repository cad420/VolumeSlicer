include(${PROJECT_SOURCE_DIR}/cmake/utils.cmake)

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
        ${MPI_CXX_LIBRARIES}
        ${OPENGL_LIBRARIES}
)
target_include_directories(
        MPILargeVolumeVisApp
        PRIVATE
        ${PROJECT_SOURCE_DIR}/third_party
        ${MPI_CXX_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/third_party/glad/include
)
target_compile_features(MPILargeVolumeVisApp PRIVATE cxx_std_17)

embed_shaders(
      CONFIG_FILE ${CMAKE_CURRENT_SOURCE_DIR}/shaders.hpp.in
        TARGET ${CMAKE_CURRENT_SOURCE_DIR}/shader.hpp
        DIR ${CMAKE_CURRENT_SOURCE_DIR}/shader
)