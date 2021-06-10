//
// Created by wyz on 2021/6/9.
//

#ifndef VOLUMESLICER_BLOCK_LOADER_HPP
#define VOLUMESLICER_BLOCK_LOADER_HPP

#include<VolumeSlicer/export.hpp>
#include<VolumeSlicer/define.hpp>
#include<Common/cuda_mem_pool.hpp>
#include<IO/reader_impl.hpp>
VS_START


class BlockLoader{
public:
    BlockLoader(const char* file_path);

    
private:


};

VS_END

#endif //VOLUMESLICER_BLOCK_LOADER_HPP
