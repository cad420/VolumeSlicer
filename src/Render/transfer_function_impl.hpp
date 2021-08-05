//
// Created by wyz on 2021/6/16.
//

#ifndef VOLUMESLICER_TRANSFER_FUNCTION_IMPL_HPP
#define VOLUMESLICER_TRANSFER_FUNCTION_IMPL_HPP

#include<VolumeSlicer/export.hpp>
#include<VolumeSlicer/transfer_function.hpp>
#include <map>
#include <vector>
#include <cmath>
#include <iostream>
#include <array>
#define TF_DIM 256
VS_START
    class TransferFuncImpl{
    public:
        explicit TransferFuncImpl(std::map<uint8_t ,std::array<double,4>> color_setting): color_setting(color_setting){};
        explicit TransferFuncImpl(const TransferFunc& tf){
            for(auto& it:tf.points){
                color_setting[it.key]=it.value;
            }
        }
        ~TransferFuncImpl()=default;
        auto getTransferFunction()->std::vector<float>&{
            if(transfer_func.empty())
                generateTransferFunc();
            return transfer_func;
        }
        void resetTransferFunc(std::map<uint8_t,std::array<double,4>> color_setting){
            this->color_setting=color_setting;
            transfer_func.clear();
            preint_transfer_func.clear();
        }
        auto getPreIntTransferFunc()->std::vector<float>&{
            if(preint_transfer_func.empty())
                generatePreIntTransferFunc();
            return preint_transfer_func;
        }
    private:
        void generateTransferFunc();
        void generatePreIntTransferFunc();
    private:
        std::map<uint8_t,std::array<double,4>> color_setting;
        std::vector<float> transfer_func;
        std::vector<float> preint_transfer_func;
        const int base_sampler_number=20;
        const int ratio=1;
    };

    inline void TransferFuncImpl::generateTransferFunc()
    {

        transfer_func.resize(TF_DIM*4);
        std::vector<uint8_t> keys;
        for(auto it:color_setting)
            keys.emplace_back(it.first);
        size_t size=keys.size();
        for(size_t i=0;i<keys[0];i++){
            transfer_func[i*4+0]=color_setting[keys[0]][0];
            transfer_func[i*4+1]=color_setting[keys[0]][1];
            transfer_func[i*4+2]=color_setting[keys[0]][2];
            transfer_func[i*4+3]=color_setting[keys[0]][3];
        }
        for(size_t i=keys[size-1];i<TF_DIM;i++){
            transfer_func[i*4+0]=color_setting[keys[size-1]][0];
            transfer_func[i*4+1]=color_setting[keys[size-1]][1];
            transfer_func[i*4+2]=color_setting[keys[size-1]][2];
            transfer_func[i*4+3]=color_setting[keys[size-1]][3];
        }
        for(size_t i=1;i<size;i++){
            int left=keys[i-1],right=keys[i];
            auto left_color=color_setting[left];
            auto right_color=color_setting[right];

            for(size_t j=left;j<=right;j++){
                transfer_func[j*4+0]=1.0f*(j-left)/(right-left)*right_color[0]+1.0f*(right-j)/(right-left)*left_color[0];
                transfer_func[j*4+1]=1.0f*(j-left)/(right-left)*right_color[1]+1.0f*(right-j)/(right-left)*left_color[1];
                transfer_func[j*4+2]=1.0f*(j-left)/(right-left)*right_color[2]+1.0f*(right-j)/(right-left)*left_color[2];
                transfer_func[j*4+3]=1.0f*(j-left)/(right-left)*right_color[3]+1.0f*(right-j)/(right-left)*left_color[3];
            }
        }
    }

    inline void TransferFuncImpl::generatePreIntTransferFunc()
    {
        if(transfer_func.empty())
            generateTransferFunc();
        preint_transfer_func.resize(4*TF_DIM*TF_DIM);

        float rayStep=1.0;
        for(int sb=0;sb<TF_DIM;sb++){
            for(int sf=0;sf<=sb;sf++){
                int offset=sf!=sb;
                int n=base_sampler_number+ratio*std::abs(sb-sf);
                float stepWidth=rayStep/n;
                float rgba[4]={0,0,0,0};
                for(int i=0;i<n;i++){
                    float s=sf+(sb-sf)*(float)i / n;
                    float sFrac=s-std::floor(s);
                    float opacity=(transfer_func[int(s)*4+3]*(1.0-sFrac)+transfer_func[((int)s+offset)*4+3]*sFrac)*stepWidth;
                    float temp=std::exp(-rgba[3])*opacity;
                    rgba[0]+=(transfer_func[(int)s*4+0]*(1.0-sFrac)+transfer_func[(int(s)+offset)*4+0]*sFrac)*temp;
                    rgba[1]+=(transfer_func[(int)s*4+1]*(1.0-sFrac)+transfer_func[(int(s)+offset)*4+1]*sFrac)*temp;
                    rgba[2]+=(transfer_func[(int)s*4+2]*(1.0-sFrac)+transfer_func[(int(s)+offset)*4+2]*sFrac)*temp;
                    rgba[3]+=opacity;
                }
                preint_transfer_func[(sf*TF_DIM+sb)*4+0]=preint_transfer_func[(sb*TF_DIM+sf)*4+0]=rgba[0];
                preint_transfer_func[(sf*TF_DIM+sb)*4+1]=preint_transfer_func[(sb*TF_DIM+sf)*4+1]=rgba[1];
                preint_transfer_func[(sf*TF_DIM+sb)*4+2]=preint_transfer_func[(sb*TF_DIM+sf)*4+2]=rgba[2];
                preint_transfer_func[(sf*TF_DIM+sb)*4+3]=preint_transfer_func[(sb*TF_DIM+sf)*4+3]=1.0-std::exp(-rgba[3]);
            }
        }
    }

VS_END

#endif //VOLUMESLICER_TRANSFER_FUNCTION_IMPL_HPP
