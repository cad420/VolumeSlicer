//
// Created by wyz on 2021/6/10.
//
#include "IO/reader_impl.hpp"

#include <fstream>

#include <VolumeSlicer/Utils/logger.hpp>

VS_START

std::unique_ptr<Reader> Reader::CreateReader(const char *file_name)
{
    std::unique_ptr<ReaderImpl> reader(new ReaderImpl);
    if (!file_name)
        return reader;
    sv::LodFile lod_file;
    try
    {
        lod_file.open_lod_file(file_name);
    }
    catch (const std::exception &err)
    {
        LOG_ERROR("{0}", err.what());
        return reader;
    }
    auto min_lod = lod_file.get_min_lod();
    auto max_lod = lod_file.get_max_lod();
    auto volume_space = lod_file.get_volume_space();
    for (int i = min_lod; i <= max_lod; i++)
    {
        reader->AddLodData(i, lod_file.get_lod_file_path(i).c_str());
    }
    reader->SetVolumeSpace(volume_space);
    LOG_INFO("Successfully Create Reader, min_lod({0}),max_lod({1}).", min_lod, max_lod);
    return reader;
}

void ReaderImpl::GetPacket(const std::array<uint32_t, 4> &idx, std::vector<std::vector<uint8_t>> &packet)
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
    }
    LOG_INFO("load factor for packet cache is: {0:f}", packet_cache.get_load_factor());
}

void ReaderImpl::AddLodData(int lod, const char *path)
{
    try
    {
        if (lod < 0)
        {
            LOG_ERROR("lod({0}) < 0", lod);
            return;
        }
        readers[lod] = std::make_unique<sv::Reader>(path);
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

size_t ReaderImpl::GetBlockSizeByte()
{
    try
    {
        auto header = readers.at(min_lod)->get_header();
        size_t block_length = std::pow(2, header.log_block_length);
        return block_length * block_length * block_length;
    }
    catch (const std::exception &err)
    {
        LOG_ERROR("GetBlockSizeByte: {0}.", err.what());
        return 0;
    }
}

auto ReaderImpl::GetBlockDim(int lod) const -> std::array<uint32_t, 3>
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

auto ReaderImpl::GetBlockLength() const -> std::array<uint32_t, 4>
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

auto ReaderImpl::GetFrameShape() const -> std::array<uint32_t, 2>
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

ReaderImpl::ReaderImpl() : min_lod(0x0fffffff), max_lod(0), packet_cache(500)
{
}
auto ReaderImpl::GetVolumeSpace() const -> std::array<float, 3>
{
    return volume_space;
}
void ReaderImpl::SetVolumeSpace(const std::array<float, 3> & space)
{
    this->volume_space = space;
}

VS_END