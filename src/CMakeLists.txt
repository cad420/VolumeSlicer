include(${PROJECT_SOURCE_DIR}/third_party/spdlog.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/glm.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/VolumeCompression.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/stdgpu.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/glfw.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/utils.cmake)
find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenMP)
list(
        APPEND
        VS_SRCS
        Data/slice_impl.cpp
        Data/volume_impl.cpp
        Data/block_loader.cpp
        Data/cuda_volume_cache_impl.cpp
        Data/opengl_volume_cache_impl.cpp
        Data/cpu_volume_cache_impl.cpp
        Data/chunk_cache.cpp
        Render/CUDA/SliceRender/volume_sampler_impl.cpp
        Render/CUDA/SliceRender/raw_volume_sample.cu
        Render/CUDA/SliceRender/comp_volume_sample.cu
        Render/OpenGL/opengl_sliceraw_render_impl.cpp
        Render/CUDA/VolumeRender/cuda_comp_render_impl.cu
        Render/CUDA/VolumeRender/cuda_comp_render_impl.cpp
        Render/OpenGL/opengl_comp_render_impl.cpp
        Render/CPU/cpu_comp_render_impl.cpp
        Render/CUDA/VolumeRender/cuda_raw_render_impl.cpp
        Render/CUDA/VolumeRender/cuda_raw_render_impl.cu
        Render/OpenGL/opengl_raw_render_impl.cpp
        Render/CPU/cpu_raw_render_impl.cpp
        Render/CUDA/VolumeRender/cuda_offscreen_comp_render_impl.cpp
        Render/CUDA/VolumeRender/cuda_offscreen_comp_render_impl.cu
        Render/OpenGL/opengl_offscreen_comp_render_impl.cpp
        Render/OpenGL/simple_mesh_render_impl.cpp
        IO/reader_impl.cpp
        Common/cuda_context.cpp
        Common/image.cpp
        Common/library.cpp
        Common/library_reposity.cpp
        Common/plugin_loader.cpp
        Common/memory_helper.cpp
        Data/mesh_impl.cpp
        ${PROJECT_SOURCE_DIR}/third_party/glad/src/glad.c
)
add_library(VS_CORE
#        SHARED
        ${VS_SRCS})
add_library(VS::VS_CORE ALIAS VS_CORE)
set(VS_CORE_HEADERS ${PROJECT_SOURCE_DIR}/include;${OPENGL_INCLUDE_DIR})
set_target_properties(VS_CORE PROPERTIES
        PUBLIC_HEAD "${VS_CORE_HEADERS}"
        )

target_include_directories(VS_CORE
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        ${CUDA_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/third_party
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
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
        stdgpu::stdgpu
        glfw

        )

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/build)


if(LINUX)
    target_link_libraries(VS_CORE PRIVATE stdc++fs)
    target_compile_features(VS_CORE PRIVATE cxx_std_14)
elseif(WIN32)
    target_compile_features(VS_CORE PRIVATE cxx_std_17)
endif()

embed_shaders(
        CONFIG_FILE ${CMAKE_CURRENT_SOURCE_DIR}/Render/OpenGL/shaders.hpp.in
        TARGET ${CMAKE_CURRENT_SOURCE_DIR}/Render/OpenGL/shaders.hpp
        DIR ${CMAKE_CURRENT_SOURCE_DIR}/Render/OpenGL/ShaderAssets
)

if(VS_INSTALL)

    install(
            TARGETS VS_CORE
            DESTINATION lib
    )

    install(
            DIRECTORY ${PROJECT_SOURCE_DIR}/include/VolumeSlicer
            DESTINATION include
    )
#    install(
#            EXPORT VS-targets
#            FILE VS-targets.cmake
#            NAMESPACE VS::
#            DESTINATION lib/cmake/VS
#    )



endif()