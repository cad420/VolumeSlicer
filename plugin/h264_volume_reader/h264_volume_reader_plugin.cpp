//
// Created by wyz on 2021/10/8.
//

#include "h264_volume_reader_plugin.hpp"

#include <VolumeSlicer/Utils/logger.hpp>

VS_START

H264VolumeReaderPlugin::H264VolumeReaderPlugin() : min_lod(0x0fffffff), max_lod(0), packet_cache(500)
{
}

H264VolumeReaderPlugin::~H264VolumeReaderPlugin()
{
}

void H264VolumeReaderPlugin::Open(const std::string &filename)
{
    sv::LodFile lod_file;
    try
    {
        lod_file.open_lod_file(filename);
    }
    catch (const std::exception &err)
    {
        LOG_ERROR("{0}", err.what());
        return;
    }
    this->min_lod = lod_file.get_min_lod();
    this->max_lod = lod_file.get_max_lod();
    this->volume_space = lod_file.get_volume_space();
    for (int i = min_lod; i <= max_lod; i++)
    {
        AddLodData(i, lod_file.get_lod_file_path(i));
    }
    LOG_INFO("Successfully Create Reader, min_lod({0}),max_lod({1}).", min_lod, max_lod);
}

void H264VolumeReaderPlugin::AddLodData(int lod, const std::string &filename)
{
    try
    {
        if (lod < 0)
        {
            LOG_ERROR("lod({0}) < 0", lod);
            return;
        }
        readers[lod] = std::make_unique<sv::Reader>(filename.c_str());
        readers.at(lod)->read_header();
    }
    catch (const std::exception &err)
    {
        LOG_ERROR("AddLodData: {0}.", err.what());
        readers[lod] = nullptr;
        return;
    }
    this->min_lod = lod < min_lod ? lod : min_lod;
    this->max_lod = lod > max_lod ? lod : max_lod;
}

void H264VolumeReaderPlugin::GetPacket(const std::array<uint32_t, 4> &idx, std::vector<std::vector<uint8_t>> &packet)
{
    std::unique_lock<std::mutex> lk(mtx);
    if (idx[3] < min_lod || idx[3] > max_lod)
    {
        LOG_ERROR("GetPacket: out of range.");
        return;
    }
    auto data_ptr = packet_cache.get_value_ptr(idx);
    if (data_ptr == nullptr)
    {
        readers.at(idx[3])->read_packet({idx[0], idx[1], idx[2]}, packet);
        std::vector<std::vector<uint8_t>> tmp = packet;
        packet_cache.emplace_back(idx, std::move(tmp));
    }
    else
    {
        packet = *data_ptr;
        LOG_ERROR("find cached packet!!!");
    }
    LOG_INFO("load factor for packet cache is: {0:f}", packet_cache.get_load_factor());
    if (packet_cache.get_load_factor() == 1)
    {
        LOG_ERROR("cache is full!!!");
    }
}

size_t H264VolumeReaderPlugin::GetBlockSizeByte()
{
    try
    {
        auto header = readers.at(min_lod)->get_header();
        size_t block_length = std::pow(2, header.log_block_length);
        size_t voxel_size = sv::GetVoxelSize(header.voxel);
        if(voxel_size == 0){
            LOG_ERROR("ERROR: voxel_size = 0 and set to 1. This may be caused by use old Header format file but with new lib");
            voxel_size = 1;
        }
        return block_length * block_length * block_length * voxel_size;
    }
    catch (const std::exception &err)
    {
        LOG_ERROR("GetBlockSizeByte: {0}.", err.what());
        return 0;
    }
}

auto H264VolumeReaderPlugin::GetBlockLength() const -> std::array<uint32_t, 4>
{
    try
    {
        auto header = readers.at(min_lod)->get_header();
        uint32_t block_length = std::pow(2, header.log_block_length);
        return std::array<uint32_t, 4>{block_length, header.padding, (uint32_t)min_lod, (uint32_t)max_lod};
    }
    catch (const std::exception &err)
    {
        LOG_ERROR("GetBlockLength: {0}.", err.what());
        return {0, 0, 0, 0};
    }
}

auto H264VolumeReaderPlugin::GetBlockDim(int lod) const -> std::array<uint32_t, 3>
{
    try
    {
        auto header = readers.at(lod)->get_header();
        return {header.block_dim_x, header.block_dim_y, header.block_dim_z};
    }
    catch (const std::exception &err)
    {
        LOG_ERROR("GetBlockDim: {0}.", err.what());
        return {0, 0, 0};
    }
}

auto H264VolumeReaderPlugin::GetFrameShape() const -> std::array<uint32_t, 2>
{
    try
    {
        auto header = readers.at(min_lod)->get_header();
        return {header.frame_width, header.frame_height};
    }
    catch (const std::exception &err)
    {
        LOG_ERROR("GetFrameShape: {0}.", err.what());
        return {0, 0};
    }
}
auto H264VolumeReaderPlugin::GetVolumeSpace() const -> std::array<float, 3>
{
    return this->volume_space;
}

VS_END

VS_REGISTER_PLUGIN_FACTORY_IMPL(H264VolumeReaderPluginFactory)
EXPORT_PLUGIN_FACTORY_IMPL(H264VolumeReaderPluginFactory)
