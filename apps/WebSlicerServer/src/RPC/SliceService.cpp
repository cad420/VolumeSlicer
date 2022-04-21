//
// Created by wyz on 2021/10/28.
//
#include "RPC/SliceService.hpp"
#include "Dataset/CompVolume.hpp"

#include <VolumeSlicer/Render/volume_sampler.hpp>
#include <VolumeSlicer/Utils/logger.hpp>
#include <VolumeSlicer/Utils/timer.hpp>

VS_START
namespace remote
{
SliceService::SliceService() : methods(std::make_unique<RPCMethod>())
{
    methods->register_method("get",{},RPCMethod::GetHandler(&SliceService::get,*this));
    methods->register_method("render", {"slice"}, RPCMethod::GetHandler(&SliceService::render, *this));
    methods->register_method("map",{"slice"},RPCMethod::GetHandler(&SliceService::map,*this));
    methods->register_method("render_frame",{"slice"},RPCMethod::GetHandler(&SliceService::render_frame,*this));
}
void SliceService::process_message(const uint8_t *message, uint32_t size, const SliceService::Callback &callback)
{


    uint8_t *response;
    size_t total;
    mpack_writer_t response_writer;
    try
    {
        mpack_tree_t tree;
        mpack_tree_init_data(&tree,reinterpret_cast<const char*>(message),size);
        mpack_tree_parse(&tree);
        if(mpack_tree_error(&tree)!=mpack_ok){
            throw std::runtime_error("UnPack error");
        }
        mpack_node_t root = mpack_tree_root(&tree);

        if(root.data->type != mpack_type_map ||
            !mpack_node_map_contains_cstr(root,"method") ||
            !mpack_node_map_contains_cstr(root,"params")){
            throw std::runtime_error("Invalid request error: no method or params");
        }

        auto method_node = mpack_node_map_cstr(root,"method");
        if(method_node.data->type != mpack_type_str){
            throw std::runtime_error("Invalid request error: method is not str");
        }
        auto method = std::string(mpack_node_str(method_node),mpack_node_strlen(method_node));

        auto param_node = mpack_node_map_cstr(root,"params");



        mpack_writer_init_growable(&response_writer, reinterpret_cast<char **>(&response),
                                   reinterpret_cast<size_t *>(&total));
        if(mpack_node_map_contains_cstr(root,"id")){
            mpack_start_map(&response_writer, 2);
            auto id_node = mpack_node_map_cstr(root,"id");
            auto id = std::string(mpack_node_str(id_node),mpack_node_strlen(id_node));
            LOG_INFO("message id: {}",id);
            mpack_write_cstr(&response_writer,"id");
            mpack_write_cstr(&response_writer,id.c_str());
        }
        else
            mpack_start_map(&response_writer, 1);

        methods->invoke(method,param_node,&response_writer);
    }
    catch (const std::exception& err)
    {
        LOG_ERROR(err.what());
        mpack_writer_init_growable(&response_writer, reinterpret_cast<char **>(&response),
                                   reinterpret_cast<size_t *>(&total));
        mpack_start_map(&response_writer, 1);
        mpack_write_cstr(&response_writer,"error");
        mpack_write_cstr(&response_writer,err.what());
    }


    mpack_finish_map(&response_writer);
    if(mpack_writer_destroy(&response_writer)==mpack_ok){
        callback(reinterpret_cast<const uint8_t*>(response),total);
    }
    free(response);
    LOG_INFO("finish");
}
//-----------------------------------------------------

bool SliceRenderer::occupied = false;
std::mutex SliceRenderer::mtx;
std::condition_variable SliceRenderer::cv;

const auto& SliceRenderer::GetSliceRenderer()
{
    static SliceRenderer sliceRenderer;

    static auto slice_sampler = VolumeSampler::CreateVolumeSampler(
        VolumeDataSet::GetVolume()
    );
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk,[](){
        return !occupied;
    });
    occupied = true;
    return slice_sampler;
}

void SliceRenderer::Release()
{
    occupied = false;
    cv.notify_one();
}
SliceRenderer::SliceRenderer()
{
    SetCUDACtx(0);
}

static void MaxMix(std::vector<uint8_t>& res,std::vector<uint8_t>& v){
    assert(res.size()==v.size());
    for(size_t i = 0;i<res.size();i++){
        res[i] = (std::max)(res[i],v[i]);
    }
}

//rpc method
Volume SliceService::get()
{
    Volume volume;
    volume.volume_name="neuron";
    volume.volume_dim={VolumeDataSet::GetVolume()->GetVolumeDimX(),
                       VolumeDataSet::GetVolume()->GetVolumeDimY(),
                       VolumeDataSet::GetVolume()->GetVolumeDimZ()};
    volume.volume_space={VolumeDataSet::GetVolume()->GetVolumeSpaceX(),
                         VolumeDataSet::GetVolume()->GetVolumeSpaceY(),
                         VolumeDataSet::GetVolume()->GetVolumeSpaceZ()};
    return volume;
}
Image SliceService::render(Slice slice)
{
    AutoTimer timer("render");
    Timer t;
    t.start();
    cuCtxSetCurrent(GetCUDACtx());
//    static auto space_x = VolumeDataSet::GetVolume()->GetVolumeSpaceX();
//    static auto space_y = VolumeDataSet::GetVolume()->GetVolumeSpaceY();
//    static auto space_z = VolumeDataSet::GetVolume()->GetVolumeSpaceZ();
//    static auto base_space = (std::min)({space_x,space_y,space_z});
//    static auto base_sample_space = base_space * 0.5;
//    int d = depth / base_sample_space;
//    if(d*base_sample_space<depth){
//        depth = depth / d;
//    }
//    //transform to voxel
//    depth = depth / base_space;

//    auto slicer = Slicer::CreateSlicer(slice);
//    slicer->SetSliceSpaceRatio({space_x/base_space,space_y/base_space,space_z/base_space});
    const auto& slice_renderer = SliceRenderer::GetSliceRenderer();
    t.stop();
    t.print_duration();
    t.start();
    std::vector<uint8_t> res(slice.n_pixels_width*slice.n_pixels_height);
    slice_renderer->Sample(slice,res.data(),false);
//    t.stop();
//    t.print_duration();
//    t.start();
//    std::vector<uint8_t> tmp;
//    if(d>0){
//        tmp.resize(slice.n_pixels_width*slice.n_pixels_height);
//    }
//    if((direction & 0b1)){
//        for(int i=1;i<=d;i++){
//            slicer->MoveByNormal(depth);
//            {
//                AutoTimer timer("sample");
//                slice_renderer->Sample(slicer->GetSlice(), tmp.data(), false);
//            }
//            MaxMix(res,tmp);
//        }
//    }
//    t.stop();
//    t.print_duration();
//    t.start();
//    slicer->SetSlice(slice);
//    if((direction & 0b10)){
//        for(int i=1;i<=d;i++){
//            slicer->MoveByNormal(-depth);
//            slice_renderer->Sample(slicer->GetSlice(),tmp.data(),false);
//            MaxMix(res,tmp);
//        }
//    }
    t.stop();
    t.print_duration();
    t.start();
    auto img = Image::encode(res.data(),slice.n_pixels_width,slice.n_pixels_height,1,
                             Image::Format::JPEG,Image::Quality::MEDIUM);
    SliceRenderer::Release();
    t.stop();
    t.print_duration();
    return img;
}
Image SliceService::map(Slice slice)
{
    AutoTimer timer("map");
    cuCtxSetCurrent(GetCUDACtx());
    auto world_slice = slice;
    static auto raw_volume_sampler = VolumeSampler::CreateVolumeSampler(VolumeDataSet::GetRawVolume());
    static int raw_lod =6;
    static auto volume_space_x = VolumeDataSet::GetVolume()->GetVolumeSpaceX();
    static auto volume_space_y = VolumeDataSet::GetVolume()->GetVolumeSpaceY();
    static auto volume_space_z = VolumeDataSet::GetVolume()->GetVolumeSpaceZ();
    static auto raw_dim_x = VolumeDataSet::GetRawVolume()->GetVolumeDimX();
    static auto raw_dim_y = VolumeDataSet::GetRawVolume()->GetVolumeDimY();
    static auto raw_dim_z = VolumeDataSet::GetRawVolume()->GetVolumeDimZ();
    static int window_w = 500;
    static int window_h = 500;
    static float ratio = pow(2,raw_lod);
    slice.origin={slice.origin[0]/ratio,slice.origin[1]/ratio,slice.origin[2]/ratio,1.f};
    static float base_space=(std::min)({volume_space_x,volume_space_y,volume_space_z});
    static float space_ratio_x=volume_space_x/base_space;
    static float space_ratio_y=volume_space_y/base_space;
    static float space_ratio_z=volume_space_z/base_space;
    float A = slice.normal[0]*space_ratio_x;
    float B = slice.normal[1]*space_ratio_y;
    float C = slice.normal[2]*space_ratio_z;
    float length = std::sqrt(A*A+B*B+C*C);
    A /= length;
    B /= length;
    C /= length;
    float x0 = slice.origin[0];
    float y0 = slice.origin[1];
    float z0 = slice.origin[2];
    float D  = A*x0 + B*y0 + C*z0;
    static std::array<std::array<float,3>,8> pts={
        std::array<float,3>{0.0f,0.0f,0.0f},
        std::array<float,3>{raw_dim_x*1.f,0.f,0.f},
        std::array<float,3>{raw_dim_x*1.f,1.f*raw_dim_y,0.f},
        std::array<float,3>{0.f,1.f*raw_dim_y,0.f},
        std::array<float,3>{0.f,0.f,1.f*raw_dim_z},
        std::array<float,3>{1.f*raw_dim_x,0.f,1.f*raw_dim_z},
        std::array<float,3>{1.f*raw_dim_x,1.f*raw_dim_y,1.f*raw_dim_z},
        std::array<float,3>{0.f,1.f*raw_dim_y,1.f*raw_dim_z}
    };
    static std::array<std::array<int,2>,12> line_index={
        std::array<int,2>{0,1},
        std::array<int,2>{1,2},
        std::array<int,2>{2,3},
        std::array<int,2>{3,0},
        std::array<int,2>{4,5},
        std::array<int,2>{5,6},
        std::array<int,2>{6,7},
        std::array<int,2>{7,4},
        std::array<int,2>{0,4},
        std::array<int,2>{1,5},
        std::array<int,2>{2,6},
        std::array<int,2>{3,7}
    };
    int intersect_pts_cnt=0;
    float t,k;
    std::array<float,3> intersect_pts={0.f,0.f,0.f};
    std::array<float,3> tmp;
    float x1,y1,z1,x2,y2,z2;
    for(int i=0;i<line_index.size();i++){
        x1=pts[line_index[i][0]][0];
        y1=pts[line_index[i][0]][1];
        z1=pts[line_index[i][0]][2];
        x2=pts[line_index[i][1]][0];
        y2=pts[line_index[i][1]][1];
        z2=pts[line_index[i][1]][2];
        k=A*(x1-x2)+B*(y1-y2)+C*(z1-z2);
        if(std::abs(k)>0.0001f){
            t=( D - (A*x2+B*y2+C*z2) ) / k;
            if(t>=0.f && t<1.f){
                intersect_pts_cnt++;
                tmp={t*x1+(1-t)*x2,t*y1+(1-t)*y2,t*z1+(1-t)*z2};
                intersect_pts={intersect_pts[0]+tmp[0],
                               intersect_pts[1]+tmp[1],
                               intersect_pts[2]+tmp[2]};
            }
        }
    }
    intersect_pts={intersect_pts[0]/intersect_pts_cnt,
                   intersect_pts[1]/intersect_pts_cnt,
                   intersect_pts[2]/intersect_pts_cnt};
    slice.origin={intersect_pts[0],
                  intersect_pts[1],
                  intersect_pts[2],1.f};
    slice.voxel_per_pixel_width=slice.voxel_per_pixel_height=1.f;
    slice.n_pixels_width=window_w;
    slice.n_pixels_height=window_h;

    Image img;
    img.width = window_w;
    img.height = window_h;
    img.channels = 1;
    img.data.resize(window_w*window_h);
    static auto img1to3 = [](Image& image){
        assert(image.channels == 1);
        assert(image.data.size()==image.width*image.height);
        image.channels = 3;
        std::vector<uint8_t> data(image.data.size()*3);
        for(int i =0;i<image.data.size();i++){
            data[i*3+0] = image.data[i];
            data[i*3+1] = image.data[i];
            data[i*3+2] = image.data[i];
        }
        image.data = std::move(data);
    };
    static auto setPixelColor = [](Image&image,int row,int col,uint8_t r,uint8_t g,uint8_t b){
        assert(image.channels==3);
        int c = image.channels;
        int offset = (row*image.width+col) * c;

        image.data[offset+0] = r;
        image.data[offset+1] = g;
        image.data[offset+2] = b;
    };

    raw_volume_sampler->Sample(slice,img.data.data(),true);

    img1to3(img);

    float p =world_slice.voxel_per_pixel_height/ratio;
    glm::vec3 right={world_slice.right[0],world_slice.right[1],world_slice.right[2]};
    glm::vec3 up={world_slice.up[0],world_slice.up[1],world_slice.up[2]};

    glm::vec3 offset={(world_slice.origin[0]/ratio-slice.origin[0])*space_ratio_x,
                      (world_slice.origin[1]/ratio-slice.origin[1])*space_ratio_y,
                      (world_slice.origin[2]/ratio-slice.origin[2])*space_ratio_z};
    float x_offset=glm::dot(right,offset);
    float y_offset=-glm::dot(up,offset);
    int min_p_x=x_offset/slice.voxel_per_pixel_width
                + slice.n_pixels_width/2 - world_slice.n_pixels_width/2*p/slice.voxel_per_pixel_width;
    int min_p_y=y_offset/slice.voxel_per_pixel_height
                + slice.n_pixels_height/2 - world_slice.n_pixels_height/2*p/slice.voxel_per_pixel_width;
    int max_p_x=x_offset/slice.voxel_per_pixel_width
                + slice.n_pixels_width/2 + world_slice.n_pixels_width/2*p/slice.voxel_per_pixel_width;
    int max_p_y=y_offset/slice.voxel_per_pixel_height
                + slice.n_pixels_height/2 + world_slice.n_pixels_height/2*p/slice.voxel_per_pixel_width;
    min_p_x=min_p_x<0?0:min_p_x;
    max_p_x=max_p_x<slice.n_pixels_width?max_p_x:slice.n_pixels_width-1;
    min_p_y=min_p_y<0?0:min_p_y;
    max_p_y=max_p_y<slice.n_pixels_height?max_p_y:slice.n_pixels_height-1;

    for(int i=min_p_x;i<=max_p_x;i++){
        setPixelColor(img,min_p_y,i,255,0,0);
        setPixelColor(img,max_p_y,i,255,0,0);
    }
    for(int i =min_p_y;i<=max_p_y;i++){
        setPixelColor(img,i,min_p_x,255,0,0);
        setPixelColor(img,i,max_p_x,255,0,0);
    }
    return Image::encode(img.data.data(),img.width,img.height,3,
                         Image::Format::JPEG,Image::Quality::MEDIUM);
}

Frame SliceService::render_frame(Slice slice){
    Frame frame;
    frame.ret = std::move(render(slice));
    frame.map = std::move(map(slice));
    return frame;
}

}
VS_END
