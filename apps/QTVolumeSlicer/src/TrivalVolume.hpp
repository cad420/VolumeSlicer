//
// Created by wyz on 2021/8/3.
//
#pragma once
#include <iostream>
#include <spdlog/spdlog.h>
enum TVoxelType { UInt8, Float32 };
enum TVoxelFormat { Grayscale, RGB, RGBA };
struct VolumeFormat
{
    TVoxelFormat fmt;
    TVoxelType type;
    VolumeFormat() :fmt(Grayscale), type(UInt8) {}
};
class TrivalVolume{
    VolumeFormat m_fmt;
    std::unique_ptr<unsigned char[]> m_data;
    std::unique_ptr<double[]> m_isoStat;
    std::size_t m_bytes;
    int m_xSize, m_ySize, m_zSize;
    double m_maxIsoValue;
public:
    TrivalVolume(const void * data,
           size_t xSize,
           size_t ySize,
           size_t zSize,
           const VolumeFormat & fmt = VolumeFormat()
    );
    virtual ~TrivalVolume(){
        spdlog::info("Call ~TrivalVolume destructor.");
        m_data.reset();
        m_isoStat.reset();
    }
    TrivalVolume(const TrivalVolume & vol);
    TrivalVolume & operator=(const TrivalVolume & vol);

    TrivalVolume(TrivalVolume && vol)noexcept;
    TrivalVolume & operator=(TrivalVolume && vol)noexcept;
    int xLength()const;
    int yLength()const;
    int zLength()const;

    /**
     * \brief  Returns a histogram of this volume data. The histogram is made of 256 bins.
     */
    double * isoStat()const { return m_isoStat.get(); }
    double maxIsoValue()const { return m_maxIsoValue; }
    const void * data()const;
    const VolumeFormat & format()const;
    void blend(int xpos, int ypos, int zpos,void * data, size_t xlen, size_t ylen, size_t zlen,VolumeFormat sourceVolumeFormat)=delete;
private:
    void calcIsoStat();
    template<typename T>
    void normalized(T * d, int channel);
};
template <typename T>
void TrivalVolume::normalized(T* d, int channel)
{
    T minValue, maxValue;
    for(int z = 0 ; z < m_zSize;z++)
    {
        for(int y = 0;y<m_ySize;y++)
        {
            for(int x = 0 ;x<m_zSize;x++)
            {

            }
        }
    }
}

inline int TrivalVolume::xLength() const { return m_xSize; }
inline int TrivalVolume::yLength() const { return m_ySize; }
inline int TrivalVolume::zLength() const { return m_zSize; }
inline const void * TrivalVolume::data() const { return m_data.get(); }
inline const VolumeFormat & TrivalVolume::format()const { return m_fmt; }


inline TrivalVolume::TrivalVolume(const void * data, size_t xSize, size_t ySize, size_t zSize, const VolumeFormat& fmt) :
        m_xSize(xSize)
        , m_ySize(ySize)
        , m_zSize(zSize)
        , m_fmt(fmt)
        , m_data(nullptr)
        , m_isoStat(nullptr)
        , m_bytes(0)
{
    auto voxelChannel = 0;
    switch (m_fmt.fmt)
    {
        case TVoxelFormat::Grayscale:voxelChannel = 1; break;
        case TVoxelFormat::RGB:voxelChannel = 3; break;
        case TVoxelFormat::RGBA:voxelChannel = 4; break;
    }
    //size_t bytes = 0;

    switch (m_fmt.type)
    {
        case TVoxelType::UInt8:
        {
            const auto d = new unsigned char[xSize*ySize*zSize*voxelChannel];


            m_data.reset(reinterpret_cast<unsigned char*>(d));
            m_isoStat.reset(new double[256]);
            m_bytes = xSize * ySize*zSize * sizeof(unsigned char)*voxelChannel;
        }

            break;
        case TVoxelType::Float32:
        {
            const auto d = new float[xSize*ySize*zSize*voxelChannel];
            m_data.reset(reinterpret_cast<unsigned char*>(d));
            m_isoStat.reset(new double[256]);
            m_bytes = xSize * ySize * zSize * sizeof(float)*voxelChannel;
        }

            break;
    }

    if (m_data != nullptr)
    {
        std::memcpy(m_data.get(), data, m_bytes);
    }

    if (m_isoStat != nullptr)
        calcIsoStat();
}

inline TrivalVolume::TrivalVolume(const TrivalVolume& vol)
{
    *this = vol;
}
inline TrivalVolume& TrivalVolume::operator=(const TrivalVolume& vol)
{
    m_fmt = vol.m_fmt;
    m_xSize = vol.m_xSize;
    m_ySize = vol.m_ySize;
    m_zSize = vol.m_zSize;
    m_maxIsoValue = vol.m_maxIsoValue;
    m_bytes = vol.m_bytes;

    m_data.reset(new unsigned char[m_bytes]);
    m_isoStat.reset(new double[256]);
    memcpy(m_data.get(), vol.m_data.get(), m_bytes);
    memcpy(m_isoStat.get(), vol.m_isoStat.get(), 256 * sizeof(double));
    return *this;
}
inline TrivalVolume::TrivalVolume(TrivalVolume&& vol)noexcept
{
    *this = std::move(vol);
}
inline TrivalVolume& TrivalVolume::operator=(TrivalVolume&& vol)noexcept
{
    m_fmt = vol.m_fmt;
    m_xSize = vol.m_xSize;
    m_ySize = vol.m_ySize;
    m_zSize = vol.m_zSize;
    m_maxIsoValue = vol.m_maxIsoValue;
    m_bytes = vol.m_bytes;

    m_data = std::move(vol.m_data);
    m_isoStat = std::move(vol.m_isoStat);

    return *this;
}
inline void TrivalVolume::calcIsoStat()
{
    memset(m_isoStat.get(), 0, sizeof(double) * 256);

    int size=m_zSize*m_ySize*m_xSize;
    for(int i=0;i<size;i++){
        m_isoStat[m_data.get()[i]]+=1.0;
    }

    m_maxIsoValue = m_isoStat[0];
    for (int i = 1; i < 256; ++i){
        m_maxIsoValue = std::max(m_maxIsoValue, m_isoStat[i]);
    }
}

