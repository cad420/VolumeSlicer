//
// Created by wyz on 2021/9/30.
//

#include "raw_volume_reader_plugin.hpp"

#include <VolumeSlicer/Utils/logger.hpp>

template <class Ty>
static void LoadRawVolumeData(std::ifstream &in, std::vector<uint8_t> &volume_data)
{
    if (!in.is_open())
    {
        throw std::runtime_error("LoadRawVolumeData pass ifstream not opened");
    }
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<Ty> read_data;
    read_data.resize(file_size, 0);
    in.read(reinterpret_cast<char *>(read_data.data()), file_size);

    Ty min_value = std::numeric_limits<Ty>::max();
    Ty max_value = std::numeric_limits<Ty>::min();
    LOG_INFO("Type({0}) max value is {1}, min value is {2}.", typeid(Ty).name(), min_value, max_value);
    auto min_max = std::minmax_element(read_data.cbegin(), read_data.cend());
    min_value = *min_max.first;
    max_value = *min_max.second;
    LOG_INFO("Read volume data max value is {0}, min value is {1}.", max_value, min_value);
    volume_data.resize(file_size, 0);
    for (size_t i = 0; i < volume_data.size(); i++)
    {
        volume_data[i] = 1.f * (read_data[i] - min_value) / (max_value - min_value) * 255;
    }
}

template <> static
void LoadRawVolumeData<uint8_t>(std::ifstream &in, std::vector<uint8_t> &volume_data)
{
    if (!in.is_open())
    {
        throw std::runtime_error("LoadRawVolumeData pass ifstream not opened");
    }
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    volume_data.resize(file_size, 0);
    in.read(reinterpret_cast<char *>(volume_data.data()), file_size);
    LOG_INFO("Finish read Uint8 volume.");
}

VS_START

void RawVolumeReaderPlugin::Open(const std::string &filename, vs::VoxelType type, const std::array<uint32_t, 3> &dim)
{
    in.open(filename, std::ios::binary);
    if (!in.is_open())
    {
        throw std::runtime_error("Can't open file: " + filename);
    }
    this->dim = dim;
    this->voxel_type = type;
}

void RawVolumeReaderPlugin::GetData(std::vector<uint8_t> &data)
{
    size_t expect_size = (size_t)dim[0] * dim[1] * dim[2];
    switch (voxel_type)
    {
    case VoxelType::UInt8:
        LoadRawVolumeData<uint8_t>(in, data);
        break;
    case VoxelType::UInt16:
        LoadRawVolumeData<uint16_t>(in, data);
        break;
    case VoxelType::UInt32:
        LoadRawVolumeData<uint32_t>(in, data);
        break;
    }
    if (data.size() != expect_size)
    {
        throw std::logic_error("Read raw volume size is not equal to expect size calculated by dim");
    }
}

void RawVolumeReaderPlugin::Close()
{
    in.close();
}

RawVolumeReaderPlugin::~RawVolumeReaderPlugin()
{
    RawVolumeReaderPlugin::Close();
}

VS_END

VS_REGISTER_PLUGIN_FACTORY_IMPL(RawVolumeReaderPluginFactory)
EXPORT_PLUGIN_FACTORY_IMPL(RawVolumeReaderPluginFactory)