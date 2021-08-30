//
// Created by wyz on 2021/8/26.
//

#pragma once

#include <VolumeSlicer/export.hpp>
#include <array>
#include <cstdint>
#include <functional>
VS_START

template <typename TexType, uint32_t DIM>
class TextureBase
{
    struct UnInit
    {
    };

  public:
    friend class TextureFile;
    static_assert(DIM <= 3, "Max DIM is 3");
    static_assert(DIM > 0, "DIM should > 0");
    using Texel = TexType;
    using Self = TextureBase<TexType, DIM>;
    using SizeType = size_t;
    using CoordType = std::array<uint32_t, DIM>;
    static constexpr uint32_t dim = DIM;

    TextureBase() noexcept : data(nullptr), num(0), shape({0})
    {
    }

    TextureBase(const CoordType &coord, const Texel &init_value = Texel()) : TextureBase(coord, UnInit())
    {
        for (size_t i = 0; i < num; i++)
            new (data + i) Texel(init_value);
    }
    TextureBase(const CoordType &coord, UnInit) : shape(coord), num(1)
    {
        for (int i = 0; i < dim; i++)
        {
            num *= coord[i];
        }
        data = static_cast<Texel *>(::operator new(sizeof(Texel) * num));
    }
    TextureBase(const Self &other) : num(other.num), shape(other.shape)
    {
        if (other.IsAvailable())
        {
            data = static_cast<Texel *>(::operator new(sizeof(Texel) * num));
            for (size_t i = 0; i < num; i++)
            {
                new (data + i) Texel(other.data[i]);
            }
        }
        else
            ReleaseData();
    }
    TextureBase(Self &&other) noexcept : num(other.num), shape(other.shape), data(other.data)
    {
        other.data = nullptr;
    }
    Self &operator=(const Self &other)
    {
        if (IsAvailable())
            Destroy();
        new (this) Self(other);
        return *this;
    }
    Self &operator=(Self &&other) noexcept
    {
        if (IsAvailable())
            Destroy();
        new (this) Self(std::move(other));
        return *this;
    }
    ~TextureBase()
    {
        if (IsAvailable())
            ReleaseData();
    }
    Self &GetBase()
    {
        return *this;
    }
    const Self &GetBase() const
    {
        return *this;
    }
    bool IsAvailable() const noexcept
    {
        return data != nullptr;
    }
    void Destroy()
    {
        if (IsAvailable())
        {
            ReleaseData();
        }
        num = 0;
        shape = {0};
    }
    const Texel &operator()(const CoordType &coord) const noexcept
    {
        return data[ToLinearIndex<dim>(coord)];
    }
    Texel &operator()(const CoordType &coord) noexcept
    {
        return data[ToLinearIndex<dim>(coord)];
    }
    const Texel &operator()(const SizeType &idx) const noexcept
    {
        return data[idx];
    }
    Texel &operator()(const SizeType &idx) noexcept
    {
        return data[idx];
    }

    const Texel &Fetch(const CoordType &coord) const noexcept
    {
        return (*this)(coord);
    }
    Texel &Fetch(const CoordType &coord) noexcept
    {
        return (*this)(coord);
    }

    const Texel &At(const CoordType &coord) const
    {
        auto idx = ToLinearIndex<dim>(coord);
        if (idx >= num)
            throw std::out_of_range("Texture At out of range");
        return (*this)(idx);
    }

    Texel &At(const CoordType &coord)
    {
        auto idx = ToLinearIndex<dim>(coord);
        if (idx >= num)
            throw std::out_of_range("Texture At out of range");
        return (*this)(idx);
    }
    const CoordType &GetShape() const noexcept
    {
        return shape;
    }
    SizeType GetSize() const noexcept
    {
        return num;
    }
    Texel *RawData() noexcept
    {
        return data;
    }
    const Texel *RawData() const noexcept
    {
        return data;
    }

  private:
    template <uint32_t D> SizeType ToLinearIndex(const CoordType &coord) const noexcept;

    template <> SizeType ToLinearIndex<1>(const CoordType &coord) const noexcept
    {
        return coord[0];
    }
    template <> SizeType ToLinearIndex<2>(const CoordType &coord) const noexcept
    {
        return shape[0] * coord[1] + coord[0];
    }
    template <> SizeType ToLinearIndex<3>(const CoordType &coord) const noexcept
    {
        // 3-dimension texture may overflow use uint32_t for multiply
        return (SizeType)shape[0] * (shape[1] * coord[2] + coord[1]) + coord[0];
    }
    void ReleaseData()
    {
        ::operator delete(data);
        data = nullptr;
    }

  private:
    Texel *data;
    SizeType num;
    CoordType shape;
};

template <typename TexType>
class Texture1D : public TextureBase<TexType, 1>
{
  public:
    using Base = TextureBase<TexType, 1>;
    using Texel = TexType;
    using Self = Texture1D<TexType>;
    using SizeType = size_t;
    using CoordType = std::array<uint32_t, 1>;

    Texture1D() = default;
    ~Texture1D()=default;

    Texture1D(Base&& other) noexcept
    :Base(std::move(other))
    {

    }

    explicit Texture1D(uint32_t shape, const Texel &init_value = Texel()) : Base({shape}, init_value)
    {
    }
    explicit Texture1D(const Self &) = default;
    Self &operator=(const Self &) = default;
    Texture1D(Self &&other) noexcept : Base(std::move(other.GetBase()))
    {
    }
    Self &operator=(Self &&other) noexcept
    {
        Base::GetBase() = std::move(other.GetBase());
        return *this;
    }

    uint32_t GetLength() const noexcept
    {
        return Base::GetShape()[0];
    }
    const Texel &operator()(uint32_t idx) const noexcept
    {
        return Base::operator()(CoordType{idx});
    }
    Texel &operator()(uint32_t idx) noexcept
    {
        return Base::operator()(CoordType{idx});
    }
    const Texel &Fetch(uint32_t idx) const noexcept
    {
        return Base::Fetch(CoordType{idx});
    }
    Texel &Fetch(uint32_t idx) noexcept
    {
        return Base::Fetch(CoordType{idx});
    }
    const Texel &At(uint32_t idx) const noexcept(false)
    {
        return Base::At(CoordType{idx});
    }
    Texel &At(uint32_t idx) noexcept(false)
    {
        return Base::At(CoordType{idx});
    }
    void UpdateTextureData(TexType* data,SizeType size) noexcept(::memcpy){
        if(!Base::IsAvailable()){
            return ;
        }
        size=(std::min)(size,this->GetLength());
        ::memcpy(Base::RawData(),data,size*sizeof(TexType));
    }
    void UpdateTextureSubData(SizeType offset,SizeType length,TexType* data) noexcept(::memcpy){
        if(!Base::IsAvailable()){
            return ;
        }
        if(offset >= GetLength()){
            return ;
        }
        length=(std::min)(length,GetLength()-offset);
        ::memcpy(Base::RawData()+offset,data,length*sizeof(TexType));
    }
};

template <typename TexType> class Texture2D : public TextureBase<TexType, 2>
{
  public:
    using Base = TextureBase<TexType, 2>;
    using Texel = TexType;
    using Self = Texture2D<TexType>;
    using SizeType = size_t;
    using CoordType = std::array<uint32_t, 2>;

    Texture2D() = default;
    ~Texture2D()=default;

    Texture2D(Base&& other) noexcept
        :Base(std::move(other))
    {

    }

    explicit Texture2D(CoordType shape, const Texel &init_value = Texel()) : Base(shape, init_value)
    {
    }
    explicit Texture2D(const Self &) = default;
    Self &operator=(const Self &) = default;
    Texture2D(Self &&other) noexcept : Base(std::move(other.GetBase()))
    {
    }
    Self &operator=(Self &&other) noexcept
    {
        Base::GetBase() = std::move(other.GetBase());
        return *this;
    }

    uint32_t GetWidth() const noexcept
    {
        return Base::GetShape()[0];
    }
    uint32_t GetHeight() const noexcept
    {
        return Base::GetShape()[1];
    }
    const Texel &operator()(uint32_t x, uint32_t y) const noexcept
    {
        return Base::operator()(CoordType{x, y});
    }
    Texel &operator()(uint32_t x, uint32_t y) noexcept
    {
        return Base::operator()(CoordType{x, y});
    }
    const Texel &Fetch(uint32_t x, uint32_t y) const noexcept
    {
        return Base::Fetch(CoordType{x, y});
    }
    Texel &Fetch(uint32_t x, uint32_t y) noexcept
    {
        return Base::Fetch(CoordType{x, y});
    }
    const Texel &At(uint32_t x, uint32_t y) const noexcept(false)
    {
        return Base::At(CoordType{x, y});
    }
    Texel &At(uint32_t x, uint32_t y) noexcept(false)
    {
        return Base::At(CoordType{x, y});
    }
    void UpdateTextureData(TexType* data,SizeType width,SizeType height) noexcept(::memcpy){
        if(!Base::IsAvailable()){
            return ;
        }
        width=(std::min)(width,GetWidth());
        height=(std::min)(height,GetHeight());
        ::memcpy(Base::RawData(),data,sizeof(TexType)*width*height);

    }
    void UpdateTextureSubData(SizeType offset_x,SizeType offset_y,SizeType length_x,SizeType length_y,TexType* data) noexcept(::memcpy){
        if(!Base::IsAvailable()){
            return ;
        }
        if(offset_x>=GetWidth() || offset_y>=GetHeight()){
            return ;
        }
        length_x=(std::min)(length_x,GetWidth()-offset_x);
        length_y=(std::min)(length_y,GetHeight()-offset_y);
        //todo use multi-threads to speed up
        for(SizeType row=offset_y;row<offset_y+length_y;row++){
            ::memcpy(Base::RawData()+row*GetWidth()+offset_x,
                     data+(row-offset_y)*length_x,
                     length_x*sizeof(TexType));
        }
    }
};

template <typename TexType> class Texture3D : public TextureBase<TexType, 3>
{
  public:
    using Base = TextureBase<TexType, 3>;
    using Texel = TexType;
    using Self = Texture3D<TexType>;
    using SizeType = size_t;
    using CoordType = std::array<uint32_t, 3>;

    Texture3D() = default;
    ~Texture3D()=default;

    Texture3D(Base&& other) noexcept
        :Base(std::move(other))
    {

    }

    explicit Texture3D(CoordType shape, const Texel &init_value = Texel()) : Base(shape, init_value)
    {
    }
    explicit Texture3D(const Self &) = default;
    Self &operator=(const Self &) = default;
    Texture3D(Self &&other) noexcept : Base(std::move(other.GetBase()))
    {
    }
    Self &operator=(Self &&other) noexcept
    {
        Base::GetBase() = std::move(other.GetBase());
        return *this;
    }

    uint32_t GetXSize() const noexcept
    {
        return Base::GetShape()[0];
    }
    uint32_t GetYSize() const noexcept
    {
        return Base::GetShape()[1];
    }
    uint32_t GetZSize() const noexcept
    {
        return Base::GetShape()[2];
    }
    const Texel &operator()(uint32_t x, uint32_t y, uint32_t z) const noexcept
    {
        return Base::operator()(CoordType{x, y, z});
    }
    Texel &operator()(uint32_t x, uint32_t y, uint32_t z) noexcept
    {
        return Base::operator()(CoordType{x, y, z});
    }
    const Texel &Fetch(uint32_t x, uint32_t y, uint32_t z) const noexcept
    {
        return Base::Fetch(CoordType{x, y, z});
    }
    Texel &Fetch(uint32_t x, uint32_t y, uint32_t z) noexcept
    {
        return Base::Fetch(CoordType{x, y, z});
    }
    const Texel &At(uint32_t x, uint32_t y, uint32_t z) const noexcept(false)
    {
        return Base::At(CoordType{x, y, z});
    }
    Texel &At(uint32_t x, uint32_t y, uint32_t z) noexcept(false)
    {
        return Base::At(CoordType{x, y, z});
    }
    void UpdateTextureData(TexType*data, SizeType length_x,SizeType length_y,SizeType length_z) noexcept(::memcpy){
        if(!Base::IsAvailable()){
            return ;
        }
        length_x=(std::min)(length_x,GetXSize());
        length_y=(std::min)(length_y,GetYSize());
        length_z=(std::min)(length_z,GetZSize());
        ::memcpy(Base::RawData(),data,sizeof(TexType)*length_x*length_y*length_z);
    }
    void UpdateTextureSubData(SizeType offset_x,SizeType offset_y,SizeType offset_z,
                              SizeType length_x,SizeType length_y,SizeType length_z,TexType* data) noexcept(::memcpy){
        if(!Base::IsAvailable()){
            return ;
        }
        if(offset_x>= GetXSize() || offset_y >= GetYSize() || offset_z >= GetZSize()){
            return;
        }
        length_x=(std::min)(length_x,GetXSize()-offset_x);
        length_y=(std::min)(length_y,GetYSize()-offset_y);
        length_z=(std::min)(length_z,GetZSize()-offset_z);
        //todo use multi-threads to speed up
        for(SizeType depth=offset_z;depth<offset_z+length_z;depth++){
            for(SizeType row=offset_y;row<offset_y+length_y;row++){
                ::memcpy(Base::RawData()+depth*GetXSize()*GetYSize()+row*GetXSize()+offset_x,
                         data+(depth-offset_z)*length_x*length_y+(row-offset_y)*length_x,
                         length_x*sizeof(TexType));
            }
        }
    }
};

VS_END
