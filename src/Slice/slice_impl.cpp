//
// Created by wyz on 2021/6/7.
//
#include<stdexcept>

#include<spdlog/spdlog.h>
#include<glm/gtc/matrix_transform.hpp>
#include"Slice/slice_impl.hpp"

VS_START



SlicerImpl::SlicerImpl(const Slice &slice)
:ratio({1.f,1.f,1.f})
{
    SlicerImpl::SetSlice(slice);
    spdlog::info("Successfully create slicer.");
}

void SlicerImpl::SetSlice(const Slice &slice) {
    if(!IsValidSlice(slice)){
        throw std::invalid_argument("illegal slice");
    }
    this->origin={slice.origin[0],slice.origin[1],slice.origin[2]};
    this->normal=glm::normalize(glm::vec3{slice.normal[0],slice.normal[1],slice.normal[2]});
    this->up=glm::normalize(glm::vec3{slice.up[0],slice.up[1],slice.up[2]});
    this->right=glm::normalize(glm::vec3{slice.right[0],slice.right[1],slice.right[2]});
    this->n_pixels_width=slice.n_pixels_width;
    this->n_pixels_height=slice.n_pixels_height;
    this->voxel_per_pixel_width=slice.voxel_per_pixel_width;
    this->voxel_per_pixel_height=slice.voxel_per_pixel_height;

    this->image.resize((size_t)this->n_pixels_width*this->n_pixels_height,0);
    SetStatus(true);
}

bool SlicerImpl::IsValidSlice(const Slice &slice) const {
    if(slice.n_pixels_width>MAX_SLICE_W || slice.n_pixels_height>MAX_SLICE_H){
        spdlog::error("n_pixels_width(height) is too big.");
        return false;
    }
    if(slice.voxel_per_pixel_width<=0.f || slice.voxel_per_pixel_height<=0.f){
        spdlog::error("voxel_per_pixel_width(height) <= 0.f");
        return false;
    }
//    glm::vec3 _origin={slice.origin[0],slice.origin[1],slice.origin[2]};
    glm::vec3 _normal={slice.normal[0],slice.normal[1],slice.normal[2]};
    _normal=glm::normalize(_normal);
    glm::vec3 _up={slice.up[0],slice.up[1],slice.up[2]};
    _up=glm::normalize(_up);
    glm::vec3 _right={slice.right[0],slice.right[1],slice.right[2]};
    _right=glm::normalize(_right);
    float d1=glm::dot(_normal,_up);
    float d2=glm::dot(_normal,_right);
    float d3=glm::dot(_up,_right);
    if(d1>FLOAT_ZERO || d2>FLOAT_ZERO || d3>FLOAT_ZERO){
        spdlog::error("normal right up are not all dot equal zero:{0} {1} {2}.",d1,d2,d3);
        spdlog::error("normal:{0} {1} {2}.",normal[0],normal[1],normal[2]);
        spdlog::error("up:{0} {1} {2}.",up[0],up[1],up[2]);
        spdlog::error("right:{0} {1} {2}.",right[0],right[1],right[2]);
        return false;
    }
    return true;
}

void SlicerImpl::MoveByNormal(float dist) {
    spdlog::info("slice ratio {0} {1} {2}.",ratio.x,ratio.y,ratio.z);
    this->origin+=(dist*this->normal)/ratio;
    SetStatus(true);
}

void SlicerImpl::MoveInPlane(float offsetX, float offsetY) {
    spdlog::info("slice ratio {0} {1} {2}.",this->ratio.x,this->ratio.y,this->ratio.z);
    this->origin+=(offsetX*this->right+offsetY*this->up)/this->ratio;
    SetStatus(true);
}

void SlicerImpl::StretchInXY(float scaleX, float scaleY) {
    if(scaleX<=0.f || scaleY<=0.f){
        spdlog::error("scale <=0.f, not accept!");
        return;
    }
    this->voxel_per_pixel_width *=scaleX;
    this->voxel_per_pixel_height*=scaleY;
    SetStatus(true);
}

void SlicerImpl::RotateByX(float degree) {
    glm::mat4 trans(1.0);
    trans=glm::rotate(trans,degree,this->right);
    this->normal=trans*glm::vec4(this->normal,0.f);
    this->up=trans*glm::vec4(this->up,0.f);
    SetStatus(true);
}

void SlicerImpl::RotateByY(float degree) {
    glm::mat4 trans(1.0);
    trans=glm::rotate(trans,degree,this->up);
    this->normal=trans*glm::vec4(this->normal,0.f);
    this->right=trans*glm::vec4(this->right,0.f);
    SetStatus(true);
}

void SlicerImpl::RotateByZ(float degree) {
    glm::mat4 trans(1.0);
    trans=glm::rotate(trans,degree,this->normal);
    this->up=trans*glm::vec4(this->up,0.f);
    this->right=trans*glm::vec4(this->right,0.f);
    SetStatus(true);
}

void SlicerImpl::NormalX() {
    glm::mat4 trans(1.0);
    glm::vec3 x={1.f,0.f,0.f};
    if(glm::length(x-this->normal)<0.0001f) return;
    glm::vec3 axis=glm::normalize(glm::cross(x,this->normal));
    float degree=-glm::acos(glm::dot(x,this->normal));
    trans=glm::rotate(trans,degree,axis);
    this->up=trans*glm::vec4(this->up,0.f);
    this->right=trans*glm::vec4(this->right,0.f);
    this->normal=x;
    SetStatus(true);
}

void SlicerImpl::NormalY() {
    glm::mat4 trans(1.0);
    glm::vec3 x={0.f,1.f,0.f};
    if(glm::length(x-this->normal)<0.0001f) return;
    glm::vec3 axis=glm::normalize(glm::cross(x,this->normal));
    float degree=-glm::acos(glm::dot(x,this->normal));
    trans=glm::rotate(trans,degree,axis);
    this->up=trans*glm::vec4(this->up,0.f);
    this->right=trans*glm::vec4(this->right,0.f);
    this->normal=x;
    SetStatus(true);
}

void SlicerImpl::NormalZ() {
    glm::mat4 trans(1.0);
    glm::vec3 x={0.f,0.f,1.f};
    if(glm::length(x-this->normal)<0.0001f) return;
    glm::vec3 axis=glm::normalize(glm::cross(x,this->normal));
    float degree=-glm::acos(glm::dot(x,this->normal));
    trans=glm::rotate(trans,degree,axis);
    this->up=trans*glm::vec4(this->up,0.f);
    this->right=trans*glm::vec4(this->right,0.f);
    this->normal=x;
    SetStatus(true);
}

void SlicerImpl::NormalIncreaseX(float d) {
    float x0=this->normal.x;
    float y0=this->normal.y;
    float z0=this->normal.z;
    float x1=this->normal.x+d;
    float y1,z1;
    if(x1 >  1.0) x1 =  1.0;
    if(x1 < -1.0) x1 = -1.0;
    if(std::abs(x1-x0)<0.0001f) return;
    // x0*x0 + y0*y0 + z0*z0 = 1
    // x1*x1 + y1*y1 + z1*z1 = 1
    // t = y0/z0 = y1/z1
    // x1*x1 + t*t*z1*z1 + z1*z1 = 1
    // (1+t*t)*z1*z1 = 1 - x1*x1
    // z1*z1 = (1 - x1*x1) / (1 + t*t)
    if(std::abs(z0)>0.0001f){
        float t=y0/z0;
        z1=std::sqrt((1-x1*x1)/(1+t*t));
        y1=z1*t;
    }
    else if(std::abs(y0)>0.0001f){
        float t=z0/y0;
        y1=std::sqrt((1-x1*x1)/(1+t*t));
        z1=y1*t;
    }
    else{
        z1=y1=std::sqrt((1-x1*x1)/2.f);
    }

    glm::mat4 trans(1.0);
    glm::vec3 x={x1,y1,z1};
    if(glm::length(x-this->normal)<0.0001f) return;
    glm::vec3 axis=glm::normalize(glm::cross(x,this->normal));
    float degree=-glm::acos(glm::dot(x,this->normal));
    trans=glm::rotate(trans,degree,axis);
    this->up=trans*glm::vec4(this->up,0.f);
    this->right=trans*glm::vec4(this->right,0.f);
    this->normal=x;
    IsValidSlice(GetSlice());
    SetStatus(true);

}

void SlicerImpl::NormalIncreaseY(float d) {
    float x0=this->normal.x;
    float y0=this->normal.y;
    float z0=this->normal.z;
    float y1=this->normal.y+d;
    float x1,z1;
    if(y1 >  1.0) y1 =  1.0;
    if(y1 < -1.0) y1 = -1.0;
    if(std::abs(y1-y0)<0.0001f) return;

    if(std::abs(z0)>0.0001f){
        float t=x0/z0;
        z1=std::sqrt((1-y1*y1)/(1+t*t));
        x1=z1*t;
    }
    else if(std::abs(x0)>0.0001f){
        float t=z0/x0;
        x1=std::sqrt((1-y1*y1)/(1+t*t));
        z1=x1*t;
    }
    else{
        z1=x1=std::sqrt((1-y1*y1)/2.f);
    }

    glm::mat4 trans(1.0);
    glm::vec3 x={x1,y1,z1};
    if(glm::length(x-this->normal)<0.0001f) return;
    glm::vec3 axis=glm::normalize(glm::cross(x,this->normal));
    float degree=-glm::acos(glm::dot(x,this->normal));
    trans=glm::rotate(trans,degree,axis);
    this->up=trans*glm::vec4(this->up,0.f);
    this->right=trans*glm::vec4(this->right,0.f);
    this->normal=x;

    SetStatus(true);
}

void SlicerImpl::NormalIncreaseZ(float d) {
    float x0=this->normal.x;
    float y0=this->normal.y;
    float z0=this->normal.z;
    float z1=this->normal.z+d;
    float x1,y1;
    if(z1 >  1.0) z1 =  1.0;
    if(z1 < -1.0) z1 = -1.0;
    if(std::abs(z1-z0)<0.0001f) return;

    if(std::abs(x0)>0.0001f){
        float t=y0/x0;
        x1=std::sqrt((1-z1*z1)/(1+t*t));
        y1=x1*t;
    }
    else if(std::abs(y0)>0.0001f){
        float t=x0/y0;
        y1=std::sqrt((1-z1*z1)/(1+t*t));
        x1=y1*t;
    }
    else{
        x1=y1=std::sqrt((1-z1*z1)/2.f);
    }

    glm::mat4 trans(1.0);
    glm::vec3 x={x1,y1,z1};
    if(glm::length(x-this->normal)<0.0001f) return;
    glm::vec3 axis=glm::normalize(glm::cross(x,this->normal));
    float degree=-glm::acos(glm::dot(x,this->normal));
    trans=glm::rotate(trans,degree,axis);
    this->up=trans*glm::vec4(this->up,0.f);
    this->right=trans*glm::vec4(this->right,0.f);
    this->normal=x;

    SetStatus(true);
}

bool SlicerImpl::IsValidVector() const {
    float d1=glm::dot(normal,up);
    float d2=glm::dot(normal,right);
    float d3=glm::dot(up,right);
    if(d1>FLOAT_ZERO || d2>FLOAT_ZERO || d3>FLOAT_ZERO){
        spdlog::error("normal right up are not all dot equal zero.");
        return false;
    }
    return true;
}

Slice SlicerImpl::GetSlice() const {
    return Slice{
        std::array<float,4>{origin.x,origin.y,origin.z,1.f},
        std::array<float,4>{normal.x,normal.y,normal.z,0.f},
        std::array<float,4>{up.x,up.y,up.z,0.f},
        std::array<float,4>{right.x,right.y,right.z,0.f},
        n_pixels_width,
        n_pixels_height,
        voxel_per_pixel_width,
        voxel_per_pixel_height
    };
}

uint32_t SlicerImpl::GetImageW() const {
    return this->n_pixels_width;
}

uint32_t SlicerImpl::GetImageH() const {
    return this->n_pixels_height;
}

uint8_t *SlicerImpl::GetImageData() {
    return image.data();
}

bool SlicerImpl::IsModified() const {
    return is_modified;
}

void SlicerImpl::SetStatus(bool modified) {
    this->is_modified=modified;
}

void SlicerImpl::SetSliceSpaceRatio(const std::array<float, 3> &ratio) {
    this->ratio={ratio[0],ratio[1],ratio[2]};
    spdlog::info("set slice ratio {0} {1} {2}.",this->ratio.x,this->ratio.y,this->ratio.z);
}


    std::unique_ptr<Slicer> Slicer::CreateSlicer(const Slice& slice) noexcept{
try{
    return std::make_unique<SlicerImpl>(slice);
}
catch (const std::exception& err) {
    spdlog::error("CreateSlicer constructor error: {0}",err.what());
    return std::unique_ptr<SlicerImpl>(nullptr);
}
}

VS_END


