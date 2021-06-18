include(${PROJECT_SOURCE_DIR}/third_party/spdlog.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/glm.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/VolumeCompression.cmake)
find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
list(
        APPEND
        VS_SRCS
        Slice/slice_impl.cpp
        Volume/volume_impl.cpp
        Volume/block_loader.cpp
        Algorithm/volume_sampler_impl.cpp
        Algorithm/raw_volume_sample.cu
        Algorithm/comp_volume_sample.cu
        Render/render_impl.cpp
        IO/reader_impl.cpp
        ${PROJECT_SOURCE_DIR}/third_party/glad_wgl/src/gl.c
        ${PROJECT_SOURCE_DIR}/third_party/glad_wgl/src/wgl.c
)
add_library(VS_CORE ${VS_SRCS})


set_target_properties(VS_CORE PROPERTIES PUBLIC_HEAD
        ${PROJECT_SOURCE_DIR}/include
        ${OPENGL_INCLUDE_DIR}
        )

target_include_directories(VS_CORE
        PUBLIC
        ${CUDA_INCLUDE_DIRS}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/third_party/glad_wgl/include

        )

target_link_libraries(VS_CORE
        PUBLIC

        spdlog::spdlog

        PRIVATE

        glm::glm
        ${CUDA_CUDA_LIBRARY}
        voxel_uncompress
        ${OPENGL_LIBRARIES}
        ${CMAKE_DL_LIBS}
        )

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/build)

install(
        TARGETS VS_CORE
        EXPORT vs-targets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include
)
target_compile_features(VS_CORE PRIVATE cxx_std_17)