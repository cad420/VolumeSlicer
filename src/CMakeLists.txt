include(${PROJECT_SOURCE_DIR}/third_party/spdlog.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/glm.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/VolumeCompression.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/stdgpu.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/utils.cmake)
find_package(CUDA REQUIRED)

find_package(OpenMP)
file(GLOB VS_CORE_SRCS
        "Common/*.cpp" "Common/*.c" "Common/*.hpp" "Common/*.h"
        "Data/*.cpp" "Data/*.hpp"
        "IO/*.cpp" "IO/*.hpp"
        "Render/CPU/*.cpp" "Render/CPU/*.hpp"
        "Render/CUDA/*.cpp" "Render/CUDA/*.hpp"
        "Render/CUDA/SliceRender/*.cu" "Render/CUDA/SliceRender/*.cuh" "Render/CUDA/SliceRender/*.cpp" "Render/CUDA/SliceRender/*.hpp"
        "Render/CUDA/VolumeRender/*.cu" "Render/CUDA/VolumeRender/*.cuh" "Render/CUDA/VolumeRender/*.cpp" "Render/CUDA/VolumeRender/*.hpp"
        )


add_library(VS_CORE
#        SHARED
        ${VS_CORE_SRCS} )

add_library(VS::VS_CORE ALIAS VS_CORE)
set(VS_CORE_HEADERS ${PROJECT_SOURCE_DIR}/include)
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
        ${CMAKE_DL_LIBS}
        OpenMP::OpenMP_CXX
        stdgpu::stdgpu
        )

if(VS_BUILD_OPENGL_RENDERER)
    include(${PROJECT_SOURCE_DIR}/third_party/glfw.cmake)
    find_package(OpenGL REQUIRED)
    file(GLOB VS_CORE_OPENGL_RENDERER_SRCS
            "Render/OpenGL/*.cpp" "Render/OpenGL/*.hpp"
            ${PROJECT_SOURCE_DIR}/third_party/glad/src/glad.c
            )
    target_sources(
            VS_CORE
            PRIVATE
            ${VS_CORE_OPENGL_RENDERER_SRCS}
    )
    target_link_libraries(
            VS_CORE
            PRIVATE
            glfw
            ${OPENGL_LIBRARIES}
    )
    target_include_directories(
            VS_CORE
            PRIVATE
            ${PROJECT_SOURCE_DIR}/third_party/glad/include
            )
    embed_shaders(
            CONFIG_FILE ${CMAKE_CURRENT_SOURCE_DIR}/Render/OpenGL/shaders.hpp.in
            TARGET ${CMAKE_CURRENT_SOURCE_DIR}/Render/OpenGL/shaders.hpp
            DIR ${CMAKE_CURRENT_SOURCE_DIR}/Render/OpenGL/ShaderAssets
    )
endif()


set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/build)


if(LINUX)
    target_link_libraries(VS_CORE PRIVATE stdc++fs)
    target_compile_features(VS_CORE PRIVATE cxx_std_14)
elseif(WIN32)
    target_compile_features(VS_CORE PRIVATE cxx_std_17)
endif()


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