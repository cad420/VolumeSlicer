include(${PROJECT_SOURCE_DIR}/third_party/spdlog.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/glm.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/VolumeCompression.cmake)
find_package(CUDA REQUIRED)
list(
        APPEND
        VS_SRCS
        Slice/slice_impl.cpp
        Volume/volume_impl.cpp
        Volume/block_loader.cpp
        Volume/volume_sampler_impl.cpp
        Render/render_impl.cpp
        IO/reader_impl.cpp
)
add_library(VS_CORE ${VS_SRCS})


set_target_properties(VS_CORE PROPERTIES PUBLIC_HEAD
        ${PROJECT_SOURCE_DIR}/include)

target_include_directories(VS_CORE
        PUBLIC
        ${CUDA_INCLUDE_DIRS}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}


        )

target_link_libraries(VS_CORE PRIVATE
        spdlog::spdlog
        glm::glm
        ${CUDA_CUDA_LIBRARY}
        voxel_uncompress
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