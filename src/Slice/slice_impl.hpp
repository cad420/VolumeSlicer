//
// Created by wyz on 2021/6/7.
//

#ifndef VOLUMESLICER_SLICE_IMPL_HPP
#define VOLUMESLICER_SLICE_IMPL_HPP

#include<VolumeSlicer/slice.hpp>


VS_START

class SlicerImpl: public Slicer{
public:
    SlicerImpl(const Slice&);

private:

    Slice slice;

};

VS_END

#endif //VOLUMESLICER_SLICE_IMPL_HPP
