include(${PROJECT_SOURCE_DIR}/third_party/spdlog.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/glm.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/VolumeCompression.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/stdgpu.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/glfw.cmake)
find_package(CUDA REQUIRED)
#find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenMP)
list(
        APPEND
        VS_SRCS
        Slice/slice_impl.cpp
        Volume/volume_impl.cpp
        Volume/block_loader.cpp
        Volume/cuda_volume_cache_impl.cpp
        Volume/opengl_volume_cache_impl.cpp
        Volume/cpu_volume_cache_impl.cpp
        Slice/volume_sampler_impl.cpp
        Slice/raw_volume_sample.cu
        Slice/comp_volume_sample.cu
        Render/render_impl.cpp
        Render/cuda_comp_render_impl.cu
        Render/cuda_comp_render_impl.cpp
        Render/opengl_comp_render_impl.cpp
        Render/cpu_comp_render_impl.cpp
        Render/cuda_raw_render_impl.cpp
        Render/cuda_raw_render_impl.cu
        Render/opengl_raw_render_impl.cpp
        Render/cpu_raw_render_impl.cpp
        Render/cuda_offscreen_comp_render_impl.cpp
        Render/cuda_offscreen_comp_render_impl.cu
        IO/reader_impl.cpp
        Common/cuda_context.cpp
        Common/image.cpp
        Common/library.cpp
        Common/library_reposity.cpp
        Common/plugin_loader.cpp
#        ${PROJECT_SOURCE_DIR}/third_party/glad_wgl/src/gl.c
#        ${PROJECT_SOURCE_DIR}/third_party/glad_wgl/src/wgl.c
        ${PROJECT_SOURCE_DIR}/third_party/glad/src/glad.c
)
add_library(VS_CORE
#        SHARED
        ${VS_SRCS})


set_target_properties(VS_CORE PROPERTIES PUBLIC_HEAD
        ${PROJECT_SOURCE_DIR}/include
        ${OPENGL_INCLUDE_DIR}
        )

target_include_directories(VS_CORE
        PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/third_party
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
#        ${PROJECT_SOURCE_DIR}/third_party/glad_wgl/include
        ${PROJECT_SOURCE_DIR}/third_party/glad/include
        )

target_link_libraries(VS_CORE
        PUBLIC

        spdlog::spdlog
        glm::glm
        ${CUDA_CUDA_LIBRARY}
        PRIVATE
        voxel_uncompress
        ${OPENGL_LIBRARIES}
        ${CMAKE_DL_LIBS}
        OpenMP::OpenMP_CXX
#        CUDA::nvrtc
        stdgpu::stdgpu
        glfw
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