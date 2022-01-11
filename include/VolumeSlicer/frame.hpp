//
// Created by wyz on 2021/6/15.
//

#pragma once

#include <VolumeSlicer/color.hpp>
#include <VolumeSlicer/define.hpp>
#include <VolumeSlicer/export.hpp>
#include <VolumeSlicer/status.hpp>

#include <array>
#include <cassert>
#include <stdexcept>
#include <vector>

VS_START

/**
 * @brief Simple class for image that can save encoded result in memory.
 */
class VS_EXPORT Img
{
  public:
    Image(Image&& image) noexcept{
        this->format = image.format;
        this->width = image.width;
        this->height = image.height;
        this->channels = image.channels;
        this->data = std::move(image.data);
        image.format = Img::Format::RAW;
        image.width = image.height = image.channels = 0;
    }
    Image& operator=(Image&& image) noexcept{
        new (this)(std::move(image));
        return *this;
    }

    enum class Format
    {
        RAW,
        JPEG
    };
    enum class Quality
    {
        HIGH = 90,
        MEDIUM = 70,
        LOW = 50
    };
    Format format;
    uint32_t width, height;
    uint8_t channels;
    std::vector<uint8_t> data;

  public:
    /**
     * @brief Encoding a RAW Img and return encoded Img with JPEG format, if encoding failed it will throw exception.
     */
    static Img encode(const uint8_t *data, uint32_t width, uint32_t height, uint8_t channels, Img::Format format,
                      Img::Quality quality = Img::Quality::MEDIUM, bool flip_vertically = false);
};

/**
 * @brief This class is used to save render result, just support uint8_t data, not provide encode.
 * @note Now it is replaced by template class Image and should not use it any more.
 */
class Frame
{
  public:
    [[deprecated]] Frame() = default;

    [[deprecated]] Frame(Frame &&frame) noexcept
    {
        this->width = frame.width;
        this->height = frame.height;
        this->channels = frame.channels;
        this->data = std::move(frame.data);
        frame.width = frame.height = frame.channels = 0;
    }

    [[deprecated]] Frame &operator=(Frame &&frame) noexcept
    {
        new (this)Frame(std::move(frame));
        return *this;
    }

    uint32_t width = 0;
    uint32_t height = 0;
    uint8_t channels = 0;
    std::vector<uint8_t> data;
};

/**
 * @brief This Image is suitable uint32_t like store rgba in a simple type like integer.
 * @note Wrapped Image for RGBA should use template partial specialization class Image<Color<T,len>>
 * @sa Image<Color<T,len>>
 */
template <class T>
class Image
{
  public:
    [[deprecated]] Image() = default;

    [[deprecated]] Image(Image &&image) noexcept
    {
        this->width = image.width;
        this->height = image.height;
        this->channels = image.channels;
        image.width = 0;
        image.height = 0;
        image.channels = 0;
        this->data = std::move(image.data);
    }

    [[deprecated]] Image &operator=(Image &&image) noexcept
    {
        new (this)Image(image);
        return *this;
    }


    template <typename TT,uint8_t n>
    [[deprecated]] std::array<T, n> at(int row, int col) const
    {
        static_assert(std::is_same<TT,T>::value, "type should same");
        if (row >= 0 && row < height && col >= 0 && col < width)
        {
            assert(channels == n);
            size_t idx = ((size_t)row * width + col) * n;
            std::array<T, n> value;
            for (int i = 0; i < n; i++)
                value[i] = data[idx + i];
            return value;
        }
        else
        {
            throw std::out_of_range("Image's vector out of range");
        }
    }

    template <typename TT>
    [[deprecated]] T at(int row,int col) const{
        static_assert(std::is_same<TT,T>::value, "type should same");
        if(channels!=1){
            throw std::runtime_error("this at should call only channels == 1");
        }
        if (row >= 0 && row < height && col >= 0 && col < width)
        {
            assert(channels == 1);
            size_t idx = (size_t)row * width + col;
            return data[idx];
        }
        else
        {
            throw std::out_of_range("Image's vector out of range");
        }
    }

    uint32_t width = 0;
    uint32_t height = 0;
    uint8_t channels = 0;
    std::vector<T> data = {};
};

template <typename T, int len>
class Image<Color<T, len>>
{
  public:
    struct UnInit
    {
    };

  public:
    using Self = Image<Color<T, len>>;
    using Ele = Color<T, len>;

    Image() : num(0), width(0), height(0), data(nullptr){};

    ~Image()
    {
        Destroy();
    }

    /**
     * @brief Construct Image with the init_value.
     * @note class Ele should have a copy constructor.
     */
    Image(uint32_t width, uint32_t height, const Ele &init_value = Ele{}) : Image(width, height, UnInit{})
    {
        for (size_t i = 0; i < num; i++)
            new (data + i) Ele(init_value);
    }

    /**
     * @brief Construct Image with un-initialize memory.
     */
    Image(uint32_t width, uint32_t height, UnInit) : width(width), height(height), num(width * height)
    {
        data = static_cast<Ele *>(::operator new(sizeof(Ele) * width * height));
    }

    Image(const Image &other) : width(other.width), height(other.height), num(other.num)
    {
        if (other.IsAvailable())
        {
            data = static_cast<Ele *>(::operator new(sizeof(Ele) * num));
            for (size_t i = 0; i < num; i++)
            {
                new (data + i) Ele(other.data[i]);
            }
        }
        else
            Destroy();
    }

    Self &operator=(const Self &other)
    {
        Destroy();
        new (this) Self(other);
        return *this;
    }

    Image(Image &&other) noexcept : width(other.width), height(other.height), num(other.num), data(other.data)
    {
        other.data = nullptr;
    }

    Self &operator=(Image &&other) noexcept
    {
        Destroy();
        new (this) Self(std::move(other));
        return *this;
    }

    void ReadFromMemory(const Ele *data, uint32_t width, uint32_t height)
    {
        if (!data || !width || !height)
            return;
        if (width != this->width || height != this->height)
        {
            Destroy();
            new (this) Self(width, height);
        }
        memcpy(this->data, data, sizeof(Ele) * width * height);
    }

    void SaveToFile(const char *file_name);

    Ele &Fetch(uint32_t x, uint32_t y) noexcept
    {
        return data[ToLinearIndex(x, y)];
    }

    const Ele &Fetch(uint32_t x, uint32_t y) const noexcept
    {
        return data[ToLinearIndex(x, y)];
    }

    /**
     * @brief Get data at position(x,y) and will throw exception if position is invalid
     */
    Ele &At(uint32_t x, uint32_t y)
    {
        if (x >= width || y >= height || ToLinearIndex(x, y) >= num)
            throw std::out_of_range("Image At out of range");
        return data[ToLinearIndex(x, y)];
    }

    const Ele &At(uint32_t x, uint32_t y) const
    {
        if (x >= width || y >= height || ToLinearIndex(x, y) >= num)
            throw std::out_of_range("Image At out of range");
        return data[ToLinearIndex(x, y)];
    }

    size_t ToLinearIndex(uint32_t x, uint32_t y) const
    {
        return y * width + x;
    }

    Ele &operator[](size_t idx)
    {
        return data[idx];
    }

    Ele *GetData()
    {
        return data;
    }

    const Ele *GetData() const
    {
        return data;
    }

    bool IsAvailable() const
    {
        return data != nullptr;
    };

    void Destroy()
    {
        if (IsAvailable())
        {
            ::operator delete(data);
            data = nullptr;
        }
        width = height = num = 0;
    }

    auto Width() const
    {
        return width;
    }

    auto Height() const
    {
        return height;
    }

  private:
    uint32_t num;
    uint32_t width, height;
    Ele *data;
};

template <typename T>
inline auto Image4ToImage3(const Image<Color<T,4>>& image4){
    int width = image4.Width(),height = image4.Height();
    Image<Color<T,3>> image3(width,height);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            auto color = image4.At(j, i);
            image3.At(j, i) = {color.r, color.g, color.b};
        }
    }
    return image3;
}

VS_END
