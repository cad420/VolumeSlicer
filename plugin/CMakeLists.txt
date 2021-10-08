option(BUILD_RAW_VOLUME_READER_PLUGIN "" ON)
option(BUILD_BLOCK_VOLUME_PROVIDER_PLUGIN "" ON)
option(BUILD_H264_VOLUME_READER_PLUGIN "" ON)










if(BUILD_RAW_VOLUME_READER_PLUGIN)
    add_subdirectory(raw_volume_reader)
endif()

if(BUILD_BLOCK_VOLUME_PROVIDER_PLUGIN)
    add_subdirectory(block_volume_provider)
endif()

if(BUILD_H264_VOLUME_READER_PLUGIN)
    add_subdirectory(h264_volume_reader)
endif()