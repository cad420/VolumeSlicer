list(
        APPEND
        RAW_VOLUME_READER_PLUGIN_SRC
        raw_volume_reader_plugin.cpp
)
add_library(RawVolumeReaderPlugin
        SHARED
        ${RAW_VOLUME_READER_PLUGIN_SRC}
        )
target_link_libraries(
        RawVolumeReaderPlugin
        PRIVATE
        VS_CORE
)
set_target_properties(
        RawVolumeReaderPlugin PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/plugins
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/plugins
)
