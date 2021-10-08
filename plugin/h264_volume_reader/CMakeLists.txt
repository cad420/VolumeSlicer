include(${PROJECT_SOURCE_DIR}/third_party/VolumeCompression.cmake)
list(
        APPEND
        H264_VOLUME_READER_PLUGIN_SRC
        h264_volume_reader_plugin.cpp
)
add_library(H264VolumeReaderPlugin
        SHARED
        ${H264_VOLUME_READER_PLUGIN_SRC})
target_link_libraries(
        H264VolumeReaderPlugin
        PRIVATE
        VS_CORE
        voxel_uncompress
)
set_target_properties(
        H264VolumeReaderPlugin PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/plugins
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/plugins
)