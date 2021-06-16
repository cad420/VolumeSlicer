//
// Created by wyz on 2021/6/15.
//

#ifndef VOLUMESLICER_FRAME_HPP
#define VOLUMESLICER_FRAME_HPP
#include<VolumeSlicer/export.hpp>
#include<VolumeSlicer/status.hpp>
#include<VolumeSlicer/define.hpp>
#include<array>
#include<cassert>
#include "render.hpp"

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

VS_END
#endif //VOLUMESLICER_FRAME_HPP
