//
// Created by wyz on 2021/6/7.
//
#include<stdexcept>

#include<spdlog/spdlog.h>
#include<glm/gtc/matrix_transform.hpp>
#include"Slice/slice_impl.hpp"

VS_START



SlicerImpl::SlicerImpl(const Slice &slice){
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
    spdlog::info("Successfully create slicer.");
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
    float d2=glm::dot(_normal,right);
    float d3=glm::dot(_up,_right);
    if(d1>FLOAT_ZERO || d2>FLOAT_ZERO || d3>FLOAT_ZERO){
        spdlog::error("normal right up are not all dot equal zero.");
        return false;
    }
    return true;
}

void SlicerImpl::MoveByNormal(float dist) {
    this->origin+=dist*this->normal;
    SetStatus(true);
}

void SlicerImpl::MoveInPlane(float offsetX, float offsetY) {
    this->origin+=offsetX*this->right+offsetY*this->up;
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
    glm::mat4 trans;
    trans=glm::rotate(trans,degree,this->right);
    this->normal=trans*glm::vec4(this->normal,0.f);
    this->up=trans*glm::vec4(this->up,0.f);
    SetStatus(true);
}

void SlicerImpl::RotateByY(float degree) {
    glm::mat4 trans;
    trans=glm::rotate(trans,degree,this->up);
    this->normal=trans*glm::vec4(this->normal,0.f);
    this->right=trans*glm::vec4(this->right,0.f);
    SetStatus(true);
}

void SlicerImpl::RotateByZ(float degree) {
    glm::mat4 trans;
    trans=glm::rotate(trans,degree,this->normal);
    this->up=trans*glm::vec4(this->up,0.f);
    this->right=trans*glm::vec4(this->right,0.f);
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


