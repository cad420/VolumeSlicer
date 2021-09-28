//
// Created by wyz on 2021/6/15.
//

#ifndef VOLUMESLICER_FRAME_HPP
#define VOLUMESLICER_FRAME_HPP
#include<VolumeSlicer/export.hpp>
#include<VolumeSlicer/status.hpp>
#include<VolumeSlicer/define.hpp>
#include<VolumeSlicer/color.hpp>
#include<VolumeSlicer/vec.hpp>
#include<array>
#include<cassert>
#include<vector>
#include<stdexcept>

VS_START
/**
 * just support uint8_t data, not provide encode
 */
class Frame{
public:
    Frame()=default;
    Frame(Frame&& frame)noexcept{
        *this=std::move(frame);
    }
    Frame& operator=(Frame&& frame) noexcept{
        this->width=frame.width;
        this->height=frame.height;
        this->channels=frame.channels;
        this->data=std::move(frame.data);
        frame.width=frame.height=frame.channels=0;
        assert(frame.data.size()==0);
        return *this;
    }

    uint32_t width = 0;
    uint32_t height = 0;
    uint8_t channels = 0;
    std::vector<uint8_t> data;
};



template<class T>
class Image{
public:
    Image()=default;
    Image(Image&& image)noexcept{
        *this=std::move(image);
    }
    Image& operator=(Image&& image) noexcept {
        this->width=image.width;
        this->height=image.height;
        this->channels=image.channels;
        image.width=0;
        image.height=0;
        image.channels=0;
        this->data=std::move(image.data);
        assert(image.data.size()==0);
    }


    template<uint8_t n>
    std::array<T,n> at(int row,int col) const{
        if(row>=0 && row <height && col>=0 && col<width){
            assert(channels==n);
            size_t idx=((size_t)row*width+col)*n;
            std::array<T,n> value;
            for(int i=0;i<n;i++)
                value[i]=data[idx + i];
            return value;
        }
        else{
            throw std::out_of_range("Image's vector out of range");
        }
    }

    uint32_t width = 0;
    uint32_t height = 0;
    uint8_t channels = 0;
    std::vector<T> data = {};
};


template <typename T,int len>
class Image<Vec<T,len>>{
    struct UnInit{

    };
  public:
    using Self = Image<Vec<T,len>>;
    using Ele  = Vec<T,len>;

    Image()
        :num(0),width(0),height(0),data(nullptr){};

    Image(uint32_t width,uint32_t height,const Ele& init_value=Ele{})
        :Image(width,height,UnInit())
    {
        for(size_t i=0;i<num;i++)
            new(data+i) Ele(init_value);
    }
    Image(uint32_t width,uint32_t height,UnInit)
        :width(width),height(height),num(width*height)
    {
        data=static_cast<Ele*>(::operator new(sizeof(Ele)*width*height));
    }

    Image(const Image& other)
        :width(other.width),height(other.height),num(other.num)
    {
        if(other.IsAvailable()){
            data=static_cast<Ele*>(::operator new(sizeof(Ele)*num));
            for(size_t i=0;i<num;i++){
                new(data+i) Ele(other.data[i]);
            }
        }
        else
            Destroy();
    }
    Self& operator=(const Self& other){
        Destroy();
        new(this) Self(other);
        return *this;
    }
    Image(Image&& other) noexcept
        :width(other.width),height(other.height),num(other.num),data(other.data)
    {
        other.data=nullptr;
    }
    Self& operator=(Image&& other) noexcept{
        Destroy();
        new(this) Self(std::move(other));
        return *this;
    }

    void ReadFromMemory(const Ele* data,uint32_t width,uint32_t height){
        if(!data || !width || !height)
            return;
        if(width!=this->width || height!=this->height){
            Destroy();
            new(this) Self(width,height);
        }
        memcpy(this->data,data,sizeof(Ele)*width*height);
    }
    void SaveToFile(const char* file_name);

    Ele& Fetch(uint32_t x,uint32_t y){
        return data[ToLinearIndex(x,y)];
    }
    const Ele& Fetch(uint32_t x,uint32_t y) const{
        return data[ToLinearIndex(x,y)];
    }
    Ele& At(uint32_t x,uint32_t y){
        if(x>=width || y>=height || ToLinearIndex(x,y)>=num)
            throw std::out_of_range("Image At out of range");
        return data[ToLinearIndex(x,y)];
    }
    const Ele& At(uint32_t x,uint32_t y) const{
        if(x>=width || y>=height || ToLinearIndex(x,y)>=num)
            throw std::out_of_range("Image At out of range");
        return data[ToLinearIndex(x,y)];
    }
    size_t ToLinearIndex(uint32_t x,uint32_t y) const{
        return y*width+x;
    }
    Ele& operator[](size_t idx){
        return data[idx];
    }
    Ele* GetData(){
        return data;
    }
    const Ele* GetData() const{
        return data;
    }
    bool IsAvailable() const {
        return data!=nullptr;
    };
    void Destroy(){
        if(IsAvailable()){
            ::operator delete(data);
            data=nullptr;
        }
        width=height=num=0;
    }
  private:
    uint32_t num;
    uint32_t width,height;
    Ele* data;
};

template<>
class Image<Color4b>{
    struct UnInit{

    };
  public:
    using Self=Image<Color4b>;

    Image()
    :num(0),width(0),height(0),data(nullptr){};

    Image(uint32_t width,uint32_t height,const Color4b& init_value=Color4b{})
    :Image(width,height,UnInit())
    {
        for(size_t i=0;i<num;i++)
            new(data+i) Color4b(init_value);
    }
    Image(uint32_t width,uint32_t height,UnInit)
    :width(width),height(height),num(width*height)
    {
        data=static_cast<Color4b*>(::operator new(sizeof(Color4b)*width*height));
    }

    Image(const Image& other)
    :width(other.width),height(other.height),num(other.num)
    {
        if(other.IsAvailable()){
            data=static_cast<Color4b*>(::operator new(sizeof(Color4b)*num));
            for(size_t i=0;i<num;i++){
                new(data+i) Color4b(other.data[i]);
            }
        }
        else
            Destroy();
    }
    Self& operator=(const Self& other){
        Destroy();
        new(this) Self(other);
        return *this;
    }
    Image(Image&& other) noexcept
        :width(other.width),height(other.height),num(other.num),data(other.data)
    {
        other.data=nullptr;
    }
    Self& operator=(Image&& other) noexcept{
        Destroy();
        new(this) Self(std::move(other));
        return *this;
    }

    void ReadFromMemory(const Color4b* data,uint32_t width,uint32_t height){
        if(!data || !width || !height)
            return;
        if(width!=this->width || height!=this->height){
            Destroy();
            new(this) Self(width,height);
        }
        memcpy(this->data,data,sizeof(Color4b)*width*height);
    }
    void SaveToFile(const char* file_name);

    Color4b& Fetch(uint32_t x,uint32_t y){
        return data[ToLinearIndex(x,y)];
    }
    const Color4b& Fetch(uint32_t x,uint32_t y) const{
        return data[ToLinearIndex(x,y)];
    }
    Color4b& At(uint32_t x,uint32_t y){
        if(x>=width || y>=height || ToLinearIndex(x,y)>=num)
            throw std::out_of_range("Image At out of range");
        return data[ToLinearIndex(x,y)];
    }
    const Color4b& At(uint32_t x,uint32_t y) const{
        if(x>=width || y>=height || ToLinearIndex(x,y)>=num)
            throw std::out_of_range("Image At out of range");
        return data[ToLinearIndex(x,y)];
    }
    size_t ToLinearIndex(uint32_t x,uint32_t y) const{
        return y*width+x;
    }
    Color4b& operator[](size_t idx){
        return data[idx];
    }
    Color4b* GetData(){
        return data;
    }
    const Color4b* GetData() const{
        return data;
    }
    bool IsAvailable() const {
        return data!=nullptr;
    };
    void Destroy(){
        if(IsAvailable()){
            ::operator delete(data);
            data=nullptr;
        }
        width=height=num=0;
    }
    Image<Color3b> ToImage3b() const{
        Image<Color3b> image(width,height);
        for(int i=0;i<height;i++){
            for(int j=0;j<width;j++){
                auto color=this->At(j,i);
                image.At(j,i) = {color.b,color.g,color.r};
            }
        }
        return image;
    }
  private:
    uint32_t num;
    uint32_t width,height;
    Color4b* data;
};


VS_END
#endif //VOLUMESLICER_FRAME_HPP
