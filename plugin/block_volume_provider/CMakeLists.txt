include(${PROJECT_SOURCE_DIR}/third_party/VolumeCompression.cmake)
list(
        APPEND
        BLOCK_VOLUME_PROVIDER_SRCS
        block_volume_provider_plugin.cpp
)
add_library(BlockVolumeProviderPlugin
        SHARED
        ${BLOCK_VOLUME_PROVIDER_SRCS})
target_link_libraries(
        BlockVolumeProviderPlugin
        PRIVATE
        VS_CORE
        voxel_uncompress
)
set_target_properties(
        BlockVolumeProviderPlugin PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/plugins
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/plugins
)