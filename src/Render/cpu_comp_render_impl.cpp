//
// Created by wyz on 2021/7/30.
//
#include "cpu_comp_render_impl.hpp"


#include "Render/Texture/texture_file.hpp"
#include "transfer_function_impl.hpp"
#include <Utils/box.hpp>
#include <unordered_set>
#include <glm/gtx/hash.hpp>
#include <future>
#include <thread>
#include <omp.h>
#include <Utils/timer.hpp>
VS_START
std::unique_ptr<CPUOffScreenCompVolumeRenderer> CPUOffScreenCompVolumeRenderer::Create(int w, int h) {
    return std::make_unique<CPUOffScreenCompVolumeRendererImpl>(w,h);
}

CPUOffScreenCompVolumeRendererImpl::CPUOffScreenCompVolumeRendererImpl(int w, int h)
:window_w(w),window_h(h)
{
    CPUOffScreenCompVolumeRendererImpl::resize(w,h);
}

void CPUOffScreenCompVolumeRendererImpl::SetVolume(std::shared_ptr<CompVolume> comp_volume) {
    block_cache_manager=std::make_unique<BlockCacheManager<BlockArray9b>>(16,1024,1024,1024);
    this->comp_volume=comp_volume;
    this->volume_dim_x   = comp_volume->GetVolumeDimX();
    this->volume_dim_y   = comp_volume->GetVolumeDimY();
    this->volume_dim_z   = comp_volume->GetVolumeDimZ();
    this->volume_space_x = comp_volume->GetVolumeSpaceX();
    this->volume_space_y = comp_volume->GetVolumeSpaceY();
    this->volume_space_z = comp_volume->GetVolumeSpaceZ();
    this->base_space     = (std::min)({volume_space_x,volume_space_y,volume_space_z});
    auto block_len_info  = comp_volume->GetBlockLength();
    this->block_length   = block_len_info[0];
    this->padding        = block_len_info[1];
    this->min_lod        = block_len_info[2];
    this->max_lod        = block_len_info[3];
    this->no_padding_block_length = block_length - 2 * padding;

    this->step           = this->base_space * 0.3;
}

void CPUOffScreenCompVolumeRendererImpl::SetCamera(Camera camera) {
    this->camera = camera;
}

void CPUOffScreenCompVolumeRendererImpl::SetTransferFunc(TransferFunc tf) {
    TransferFuncImpl tf_impl(tf);
    this->tf_1d=TextureFile::LoadTexture1DFromMemory(reinterpret_cast<Color4f*>(tf_impl.getTransferFunction().data()),256);
    this->tf_2d=TextureFile::LoadTexture2DFromMemory(reinterpret_cast<Color4f*>(tf_impl.getPreIntTransferFunc().data()),256,256);

}

void CPUOffScreenCompVolumeRendererImpl::render() {
    using VirtualBlockIndex = typename BlockCacheManager<BlockArray9b>::VirtualBlockIndex ;
    using VolumeBlock       = typename CompVolume::VolumeBlock;
    std::unordered_set<VirtualBlockIndex> missed_blocks;
    std::mutex mtx;
    auto AddMissedBlock=[&mtx,&missed_blocks](const VirtualBlockIndex& idx){
      std::lock_guard<std::mutex> lk(mtx);
        missed_blocks.insert(idx);
    };
    auto VirtualSample=[&AddMissedBlock,this](const Vec3d& sample_pos,double& scalar)->int{
        Vec4i virtual_block_idx;
        virtual_block_idx.x = sample_pos.x / no_padding_block_length;
        virtual_block_idx.y = sample_pos.y / no_padding_block_length;
        virtual_block_idx.z = sample_pos.z / no_padding_block_length;
        virtual_block_idx.w = 0;
        if(virtual_block_idx.x<0 || virtual_block_idx.y<0 || virtual_block_idx.z<0){
            scalar = 0.0;
            return 1;
        }
        bool cached=block_cache_manager->IsBlockDataCached(virtual_block_idx);
        if(!cached){
            AddMissedBlock(virtual_block_idx);
            scalar = 0.0;
            return 0;
        }
        auto physical_block_index=block_cache_manager->GetPhysicalBlockIndex(virtual_block_idx);
        if(!physical_block_index.IsValid()){
            LOG_ERROR("physical_block_index is invalid.");
            AddMissedBlock(virtual_block_idx);
            scalar = 0.0;
            return 0;
        }
        Vec3d physical_sample_pos;
        physical_sample_pos.x = (sample_pos.x - virtual_block_idx.x * no_padding_block_length + padding) / block_length;
        physical_sample_pos.y = (sample_pos.y - virtual_block_idx.y * no_padding_block_length + padding) / block_length;
        physical_sample_pos.z = (sample_pos.z - virtual_block_idx.z * no_padding_block_length + padding) / block_length;
        auto block_array      = block_cache_manager->GetBlock3DArray(physical_block_index.Index());
        scalar = block_array->Sample(physical_block_index.X(),physical_block_index.Y(),physical_block_index.Z(),
                            physical_sample_pos.x,physical_sample_pos.y,physical_sample_pos.z)/255.0;

        return 1;
    };

    double voxel = 1.0;
    double ka  = 0.25;
    double ks  = 0.36;
    double kd  = 0.5;
    double shininess = 100.0;

    auto PhongShading=[voxel,ka,ks,kd,shininess,&VirtualSample](Vec3d const& sample_pos,Vec3d const& diffuse_color,Vec3d const& view_direction)->Vec3d{
        Vec3d N;
        double x1,x2;
        VirtualSample(sample_pos + Vec3d(voxel,0.0,0.0),x1);
        VirtualSample(sample_pos + Vec3d(-voxel,0.0,0.0),x2);
        N.x=x1-x2;
        VirtualSample(sample_pos + Vec3d(0.0,voxel,0.0),x1);
        VirtualSample(sample_pos + Vec3d(0.0,-voxel,0.0),x2);
        N.y=x1-x2;
        VirtualSample(sample_pos + Vec3d(0.0,0.0,voxel),x1);
        VirtualSample(sample_pos + Vec3d(0.0,0.0,-voxel),x2);
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
    std::mutex image_mtx;
    auto SetPixel=[&image_mtx,this](int col,int row,Vec4d const& color)->void{
        std::lock_guard<std::mutex> lk(image_mtx);
        this->image.At(col,row)={color.r*255,color.g*255,color.b*255,255};
    };


    Image<Vec3d>   ray_directions(window_w,window_h,{0.0,0.0,0.0});//ray direction for each pixel
    Image<Vec3d>   ray_start_pos(window_w,window_h,{0.0,0.0,0.0});
    Image<Vec3d>   ray_stop_pos(window_w,window_h,{0.0,0.0,0.0});
    Image<Vec4d>   intermediate_result(window_w,window_h,{0.0,0.0,0.0,0.0});
    Box            volume_board_box({0.0,0.0,0.0},
                         {volume_dim_x*volume_space_x,volume_dim_y*volume_space_y,volume_dim_z*volume_space_z});
    std::cout<<"volume board box "<<volume_board_box<<std::endl;
    //1.generate ray_directions
    Vec3d view_pos   = {camera.pos[0],camera.pos[1],camera.pos[2]};
    Vec3d view_front = Normalize(Vec3d{camera.look_at[0]-camera.pos[0],
                                    camera.look_at[1]-camera.pos[1],
                                    camera.look_at[2]-camera.pos[2]});
    Vec3d view_right   = Normalize(Vec3d{camera.right[0],camera.right[1],camera.right[2]});
    Vec3d view_up      = Normalize(Vec3d{camera.up[0],camera.up[1],camera.up[2]});
    Vec3d volume_space = {volume_space_x,volume_space_y,volume_space_z};
    double scale     = 2.0 * tan(Radians(camera.zoom / 2)) / window_h ;
    double ratio     = 1.0 * window_w / window_h ;

    for(int row=0;row < window_h;row++){
        for(int col=0;col < window_w;col++){
            double x = (col+0.5-window_w/2.0)*scale*ratio;
            double y = (window_h/2.0-row-0.5)*scale;//window_h/2 is wrong if window_h is uint32_t

            Vec3d pixel_view_pos       = view_pos+view_front*1.0+x*view_right+y*view_up;
            Vec3d pixel_view_direction = Normalize(pixel_view_pos-view_pos);
            ray_directions.At(col,row) = pixel_view_direction;
//            this->image.At(col,row)={Clamp(pixel_view_direction.r,0.0,1.0)*255,
//                                        Clamp(pixel_view_direction.g,0.0,1.0)*255,
//            Clamp(pixel_view_direction.b,0.0,1.0)*255,
//                                        255};
        }
    }

    //2.generate ray_start_pos and ray_stop_pos according to ray_directions and box of volume
    for(int row=0;row < window_h;row++){
        for(int col=0;col < window_w;col++){
            SimpleRay pixel_ray(view_pos,ray_directions.At(col,row));
            auto intersect_t=IntersectWithAABB(volume_board_box,pixel_ray);
            if(IsIntersected(intersect_t.x,intersect_t.y)){
                if(intersect_t.x>0.0)
                    ray_start_pos.At(col,row) = pixel_ray.origin + intersect_t.x * pixel_ray.direction;
                else
                    ray_start_pos.At(col,row) = pixel_ray.origin;
                ray_stop_pos.At(col,row)  = pixel_ray.origin + intersect_t.y * pixel_ray.direction;
            }
//            auto p=ray_stop_pos.At(col,row);
//            this->image.At(col,row)={
//                p.r/11.0*255,
//                p.g/11.0*255,
//                p.b/11.0*255,255
//            };
        }
    }
    LOG_INFO("start render turn...");
    int turn=0;
    std::atomic<int>  render_finish_num = 0;

    while(++turn){
        //ray start pos will update for each pass
        //ray stop pos will not change for the render

        //if use omp, should add lock for write

        std::atomic<bool> render_finish = true;//or use w*h 2d-array

        for(int row=0;row < window_h;row++){
            if(row%(window_h/10)==0)
                LOG_INFO("turn {0} finish {1}",turn,row*1.0/window_h);
            AutoTimer timer;
//#pragma omp parallel for
            for(int col=0;col < window_w;col++){
//                if(col!=70 || row!=0) continue;
//                if(row%(window_h/10)==0 && col%(window_w/10)==0)
//                    LOG_INFO("turn {0} row {1} finish {2}",turn,row,col*1.0/window_w);
                Vec4d color              = intermediate_result.At(col,row);
                if(color.a>0.99)         continue;
                Vec3d last_ray_start_pos = ray_start_pos.At(col,row);
                Vec3d last_ray_stop_pos  = ray_stop_pos.At(col,row);
                Vec3d ray_direction      = last_ray_stop_pos-last_ray_start_pos;
                int steps                = Length(ray_direction)/step;
                ray_direction            = Normalize(ray_direction);
                Vec3d ray_pos            = last_ray_start_pos;
                double sample_scalar;
                double last_sample_scalar;
                //raycasting for the pixel
                int i = 0;
                for(;i<steps;i++){
                    //if the block is not cached
                    //1.record current ray pos and color as next pass's ray start pos
                    //2.record missed block
                    //3.break
                    int flag=VirtualSample(ray_pos / volume_space,sample_scalar);//record missed blocks in the function
                    if(flag == 0){
                        //record current ray pos and color
                        ray_start_pos.At(col,row) = ray_pos;
                        intermediate_result.At(col,row) = color;
                        render_finish = false;
                        break;
                    }
                    if(sample_scalar>0.0){
                        Vec4d sample_color=LinearSampler::Sample1D(tf_1d,sample_scalar);
                        if(sample_color.a > 0.0){

                            Vec3d shading_color = PhongShading(ray_pos/volume_space,Vec3d(sample_color),ray_direction);
//                            if(shading_color.r<0.0 || shading_color.r>1.0
//                                || shading_color.g<0.0 || shading_color.g>1.0
//                                    || shading_color.b<0.0 || shading_color.b>1.0)
//                            {
//                                LOG_ERROR("shading_color rgb <0.0 or >1.0");
//                                LOG_ERROR("sample_color: {0} {1} {2} {3}",sample_color.r,sample_color.g,sample_color.b,sample_color.a);
//                                LOG_ERROR("shading_color: {0} {1} {2}",shading_color.r,shading_color.g,shading_color.b);
//                            }
                            sample_color.x = shading_color.x;
                            sample_color.y = shading_color.y;
                            sample_color.z = shading_color.z;
                            color += sample_color * Vec4d(sample_color.a, sample_color.a, sample_color.a, 1.0) * (1.0 - color.a);
                        }
                    }
                    if(color.a>0.99){
                        break;
                    }
                    ray_pos += ray_direction * step;
                }
                if(i>=steps || color.a>0.99){
                    render_finish_num ++;
                    intermediate_result.At(col,row) = {color.r,color.g,color.b,1.0};
                    this->image.At(col,row)         = Color4b{
                                                    Clamp(color.r,0.0,1.0)*255,
                                                    Clamp(color.g,0.0,1.0)*255,
                                                    Clamp(color.b,0.0,1.0)*255,
                                                    255};
//                    SetPixel(col,row,color);
                }
                else{
                    render_finish=false;
                }
            }
        }
        //9217 v
        //9557 h
        //6009
        LOG_INFO("render finish {0}, num {1}",render_finish_num*1.0/(window_w*window_h),render_finish_num);
        //check if every pixel finish raycasting
//        if(render_finish)
//            break;
        if(missed_blocks.empty())
            break;

        //wait for finishing load missed blocks
        auto dummy_missed_blocks=missed_blocks;

        auto UploadMissedBlockData=[&dummy_missed_blocks,&missed_blocks,this](){
          for(auto& block:dummy_missed_blocks){
              assert(block.x >= 0 && block.y >= 0 && block.z >= 0 && block.w >= 0);
              VolumeBlock volume_block = this->comp_volume->GetBlock({(uint32_t)block.x,(uint32_t)block.y,(uint32_t)block.z,(uint32_t)block.w});
              if(volume_block.valid){
                  //UploadBlockData must be successful
                  block_cache_manager->UploadBlockData(std::move(volume_block));
                  missed_blocks.erase(block);
              }
          }
        };
        if(missed_blocks.size()>block_cache_manager->GetRemainPhysicalBlockNum()){
            LOG_INFO("current missed blocks num > remain_physical_blocks num");
            block_cache_manager->InitManagerResource();
            int i=0,n=block_cache_manager->GetRemainPhysicalBlockNum();
            for(auto& block:dummy_missed_blocks){
                this->comp_volume->SetRequestBlock({(uint32_t)block.x,(uint32_t)block.y,(uint32_t)block.z,(uint32_t)block.w});
                if(++i >= n){
                    break;
                }
            }
            while(block_cache_manager->GetRemainPhysicalBlockNum() > 0 && !missed_blocks.empty()){
                UploadMissedBlockData();
            }
        }
        else{
            LOG_INFO("remain physical blocks num is enough");
            for(auto& block : dummy_missed_blocks){
                assert(block.x >= 0 && block.y >= 0 && block.z >= 0 && block.w >= 0);
                this->comp_volume->SetRequestBlock({(uint32_t)block.x,(uint32_t)block.y,(uint32_t)block.z,(uint32_t)block.w});
            }
            while(!missed_blocks.empty()){
                UploadMissedBlockData();
            }
        }
//        assert(missed_blocks.empty());
        //update something?
        LOG_INFO("current turn {0}, total missed block num: {1}, after load remain missed num: {2}.",
                 turn,dummy_missed_blocks.size(),missed_blocks.size());
    }
    LOG_INFO("finish render.");
}

auto CPUOffScreenCompVolumeRendererImpl::GetFrame() -> const Image<uint32_t> & {
    return frame;
}

auto CPUOffScreenCompVolumeRendererImpl::GetImage() -> const Image<Color4b> &
{
    return image;
}

void CPUOffScreenCompVolumeRendererImpl::resize(int w, int h) {
    this->window_w=w;
    this->window_h=h;
    frame.width=w;
    frame.height=h;
    frame.data.resize((size_t)w*h);

    image=Image<Color4b>(w,h);
}

void CPUOffScreenCompVolumeRendererImpl::clear() {

}
void CPUOffScreenCompVolumeRendererImpl::SetMPIRender(MPIRenderParameter)
{

}
void CPUOffScreenCompVolumeRendererImpl::SetStep(double step, int steps)
{
    this->step=step;

}
CPUOffScreenCompVolumeRendererImpl::~CPUOffScreenCompVolumeRendererImpl()
{
    this->comp_volume.reset();
}

VS_END

