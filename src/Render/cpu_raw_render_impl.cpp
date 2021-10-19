//
// Created by wyz on 2021/7/30.
//
#include "cpu_raw_render_impl.hpp"
#include "Texture/texture_file.hpp"
#include "transfer_function_impl.hpp"
#include <VolumeSlicer/vec.hpp>
#include <Utils/logger.hpp>
#include <VolumeSlicer/cdf.hpp>
#include <omp.h>
#include <Utils/timer.hpp>
#include <Utils/box.hpp>
VS_START
CPURawVolumeRendererImpl::CPURawVolumeRendererImpl(int w, int h)
:window_w(w),window_h(h)
{
    CPURawVolumeRendererImpl::resize(w,h);
}
void CPURawVolumeRendererImpl::SetVolume(std::shared_ptr<RawVolume> raw_volume)
{
    this->volume_data_array=Linear3DArray<uint8_t>(raw_volume->GetVolumeDimX(),
                                                     raw_volume->GetVolumeDimY(),
                                                     raw_volume->GetVolumeDimZ(),
                                                     raw_volume->GetData());

    {
        CDFGenerator cdf_gen;
        cdf_gen.SetVolumeData(volume_data_array,cdf_block_length);
        {
            AutoTimer timer;
            cdf_gen.GenerateCDF();
        }
        this->cdf_map=std::move(cdf_gen.GetCDFArray());
//        for(auto x:cdf_map){
//            std::cout<<"block "<<x<<std::endl;
//        }
    }
    this->volume_data=TextureFile::LoadTexture3DFromMemory(raw_volume->GetData(),
                                                             raw_volume->GetVolumeDimX(),
                                                             raw_volume->GetVolumeDimY(),
                                                             raw_volume->GetVolumeDimZ());
    volume_dim_x=raw_volume->GetVolumeDimX();
    volume_dim_y=raw_volume->GetVolumeDimY();
    volume_dim_z=raw_volume->GetVolumeDimZ();
    space_x=raw_volume->GetVolumeSpaceX();
    space_y=raw_volume->GetVolumeSpaceY();
    space_z=raw_volume->GetVolumeSpaceZ();
    volume_board_x=volume_dim_x*space_x;
    volume_board_y=volume_dim_y*space_y;
    volume_board_z=volume_dim_z*space_z;

    cdf_dim_x = volume_dim_x / cdf_block_length;
    cdf_dim_y = volume_dim_y / cdf_block_length;
    cdf_dim_z = volume_dim_z / cdf_block_length;
}
void CPURawVolumeRendererImpl::SetCamera(Camera camera)
{
    this->camera=camera;
}
void CPURawVolumeRendererImpl::SetTransferFunc(TransferFunc tf)
{
    TransferFuncImpl tf_impl(tf);
    this->tf_1d=TextureFile::LoadTexture1DFromMemory(reinterpret_cast<Color4f*>(tf_impl.getTransferFunction().data()),256);
    this->tf_2d=TextureFile::LoadTexture2DFromMemory(reinterpret_cast<Color4f*>(tf_impl.getPreIntTransferFunc().data()),256,256);
}

void CPURawVolumeRendererImpl::render()
{
    Vec3d camera_pos = {camera.pos[0],camera.pos[1],camera.pos[2]};
    Vec3d view_front = Normalize(Vec3d{camera.look_at[0]-camera.pos[0],
                                       camera.look_at[1]-camera.pos[1],
                                       camera.look_at[2]-camera.pos[2]});
    Vec3d view_right = Normalize(Vec3d{camera.right[0],camera.right[1],camera.right[2]});
    Vec3d view_up    = Normalize(Vec3d{camera.up[0],camera.up[1],camera.up[2]});

    double scale = Radians(camera.zoom/2);
    double ratio = 1.0*window_w/window_h;
    double step  = 0.001;
    double voxel = 0.01;

    Vec3d volume_space = {space_x,space_y,space_z};
    Vec3d volume_board = {volume_board_x,volume_board_y,volume_board_z};
    Vec3d volume_dim   = {volume_dim_x,volume_dim_y,volume_dim_z};

    double ka  = 0.3;
    double ks  = 0.6;
    double kd  = 0.7;
    double shininess = 100.0;

    //sample_dir should normalized
    auto GetEmptySkipPos=[this](Vec3d const& sample_pos,const Vec3d& sample_dir){
        Vec3i cdf_block_idx=sample_pos / cdf_block_length;
        if(cdf_block_idx.x <0 || cdf_block_idx.x >= cdf_dim_x
            || cdf_block_idx.y<0 || cdf_block_idx.y >= cdf_dim_y
            || cdf_block_idx.z<0 || cdf_block_idx.z >= cdf_dim_z){
            return sample_pos;
        }
        int flat_cdf_block_idx = cdf_block_idx.z * cdf_dim_x * cdf_dim_y +
                                 cdf_block_idx.y * cdf_dim_x +
                                 cdf_block_idx.x;
        int cdf = cdf_map[flat_cdf_block_idx];
        if(cdf == 0) return sample_pos;
        auto box = ExpandBox(cdf-1,cdf_block_idx*cdf_block_length,(cdf_block_idx+1)*cdf_block_length);
        auto t   = IntersectWithAABB(box,SimpleRay(sample_pos,sample_dir));
        assert(t.x <= 0.0);
        return sample_pos + t.y * sample_dir;
    };

    auto PhongShaing=[voxel,ka,ks,kd,shininess,this](Vec3d sample_pos,Vec3d diffuse_color,Vec3d view_direction)->Vec3d{
        Vec3d N;
        double x1,x2;
        x1=LinearSampler::Sample3D(volume_data,sample_pos.x+voxel,sample_pos.y,sample_pos.z);
        x2=LinearSampler::Sample3D(volume_data,sample_pos.x-voxel,sample_pos.y,sample_pos.z);
        N.x=x1-x2;
        x1=LinearSampler::Sample3D(volume_data,sample_pos.x,sample_pos.y+voxel,sample_pos.z);
        x2=LinearSampler::Sample3D(volume_data,sample_pos.x,sample_pos.y-voxel,sample_pos.z);
        N.y=x1-x2;
        x1=LinearSampler::Sample3D(volume_data,sample_pos.x,sample_pos.y,sample_pos.z+voxel);
        x2=LinearSampler::Sample3D(volume_data,sample_pos.x,sample_pos.y,sample_pos.z-voxel);
        N.z=x1-x2;
        N=-Normalize(N);

        Vec3d L = -view_direction;
        Vec3d R = L;

        if(Dot(N,L)<0.0)
            N = -N;

        Vec3d ambient  = ka*diffuse_color;
        Vec3d specular = ks*std::pow(std::max(Dot(N,(L+R)/2.0),0.0),shininess)*Vec3d(1.0,1.0,1.0);
        Vec3d diffuse  = kd*std::max(Dot(N,L),0.0)*diffuse_color;

        return ambient + specular + diffuse;
    };
#pragma omp parallel for
    for(int row=0;row<window_h;row++){
//        LOG_INFO("start row {0}",row);
//        AutoTimer timer;
        for(int col=0;col<window_w;col++){


            double x = (2*(col+0.5)/window_w-1.0)*scale*ratio;
            double y = (1.0-2*(row+0.5)/window_h)*scale;

            Vec3d view_pos       = camera_pos+view_front*1.0+x*view_right+y*view_up;
            Vec3d view_direction = Normalize(view_pos-camera_pos);
            Vec4d color          = {0.0,0.0,0.0,0.0};
            Vec3d ray_pos        = camera_pos;

            for (int i = 0; i < 9000; i++){
                Vec3d sample_pos = ray_pos / volume_space; // space -> voxel
                if(sample_pos.x<0.0 || sample_pos.y<0.0 || sample_pos.z<0.0) break;
                //empty skip, update sample_pos in nearest non-empty block for the view_direction
                sample_pos = GetEmptySkipPos(sample_pos,Normalize(view_direction/volume_space));
                ray_pos = sample_pos *volume_space;

                sample_pos /= volume_dim;
                double sample_scalar = LinearSampler::Sample3D(volume_data, sample_pos.x, sample_pos.y, sample_pos.z);
                if (sample_scalar > 0.0){
                    Vec4d sample_color  = LinearSampler::Sample1D(tf_1d, sample_scalar / 255);
                    Vec3d shading_color = PhongShaing(sample_pos,Vec3d(sample_color),view_direction);
                    sample_color.x = shading_color.x;
                    sample_color.y = shading_color.y;
                    sample_color.z = shading_color.z;
                    if (sample_color.a > 0.0){
                        color += sample_color * Vec4d(sample_color.a, sample_color.a, sample_color.a, 1.0) * (1.0 - color.a);
                    }
                }
                if (color.a > 0.99)
                    break;
                ray_pos += step * view_direction;
            }
            image.At(col,row)=Color4b{Clamp(color.r,0.0,1.0)*255,
                                         Clamp(color.g,0.0,1.0)*255,
                                         Clamp(color.b,0.0,1.0)*255,
//                                         Clamp(color.a,0.0,1.0)*
                                             255};

        }
    }


}
auto CPURawVolumeRendererImpl::GetFrame() -> const Image<uint32_t> &
{
    return frame;
}
auto CPURawVolumeRendererImpl::GetImage() -> const Image<Color4b> &
{
    return image;
}

void CPURawVolumeRendererImpl::resize(int w, int h)
{
    frame.width=w;
    frame.height=h;
    frame.data.resize((size_t)w*h);

    image=Image<Color4b>(w,h);

}
void CPURawVolumeRendererImpl::clear()
{

}
void CPURawVolumeRendererImpl::SetMPIRender(MPIRenderParameter)
{

}
void CPURawVolumeRendererImpl::SetStep(double step, int steps)
{

}

//-------------------------------------------------------------------------------
std::unique_ptr<CPURawVolumeRenderer> CPURawVolumeRenderer::Create(int w, int h)
{
    return std::make_unique<CPURawVolumeRendererImpl>(w, h);
}

VS_END
