//
// Created by wyz on 20-12-4.
//

#ifndef VOLUMERENDER_BOUNDINGBOX_H
#define VOLUMERENDER_BOUNDINGBOX_H
#include <limits>
#include <vector>
#include <array>
#include <iostream>
#include <glm/glm.hpp>
#include <spdlog/spdlog.h>
#include <VolumeSlicer/export.hpp>
#include <VolumeSlicer/define.hpp>

VS_START
    class OBB;
    class AABB{
    public:
        AABB(){
            float min=std::numeric_limits<float>::lowest();
            float max=std::numeric_limits<float>::max();
            min_p={max,max,max};
            max_p={min,min,min};
            index={INVALID,INVALID,INVALID};
        };
        AABB(const glm::vec3& min_p,const glm::vec3& max_p,const std::array<uint32_t,4>& index)
        :min_p(min_p),max_p(max_p),index(index){}
        AABB(const glm::vec3& min_p,const glm::vec3& max_p):min_p(min_p),max_p(max_p),index({INVALID,INVALID,INVALID}){}

        AABB(const AABB& aabb){
            min_p=aabb.min_p;
            max_p=aabb.max_p;
            index=aabb.index;
        }
        AABB(const std::array<uint32_t,4>& index){
            float min=std::numeric_limits<float>::lowest();
            float max=std::numeric_limits<float>::max();
            min_p={max,max,max};
            max_p={min,min,min};
            this->index=index;
        }
        OBB convertToOBB() const;
        void Union(const glm::vec3& p){
            min_p={
                    std::fmin(min_p.x,p.x),
                    std::fmin(min_p.y,p.y),
                    std::fmin(min_p.z,p.z)
            };
            max_p={
                std::fmax(max_p.x,p.x),
                std::fmax(max_p.y,p.y),
                std::fmax(max_p.z,p.z)
            };
        }

        bool intersect(const AABB& aabb) const{
            if(fmax(min_p.x,aabb.min_p.x)>fmin(max_p.x,aabb.max_p.x)||
                    fmax(min_p.y,aabb.min_p.y)>fmin(max_p.y,aabb.max_p.y)||
                        fmax(min_p.z,aabb.min_p.z)>fmin(max_p.z,aabb.max_p.z))
                return false;
            return true;
        }
        friend std::ostream& operator<<(std::ostream& os,const AABB& aabb){
            os<<"("<<aabb.min_p.x<<" "<<aabb.min_p.y<<" "<<aabb.min_p.z<<")\t"
            <<"("<<aabb.max_p.x<<" "<<aabb.max_p.y<<" "<<aabb.max_p.z<<")"<<std::endl;
            return os;
        }
        bool operator==(const AABB& aabb) const {
//            std::cout<<"AABB =="<<std::endl;
            return (index[0]==aabb.index[0]
            && index[1]==aabb.index[1]
            && index[2]==aabb.index[2] && index[3]==aabb.index[3]);
        }
        bool operator=(const AABB& aabb){
            min_p=aabb.min_p;
            max_p=aabb.max_p;
            index=aabb.index;
        }
//        bool operator==(const std::array<uint32_t,3> index){
//            return (this->index[0]==index[0]
//            && this->index[1]==index[1]
//            && this->index[2]==index[2]);
//        }
        bool isCubic() const{
            auto x_len=max_p.x-min_p.x;
            auto y_len=max_p.y-min_p.y;
            auto z_len=max_p.z-min_p.z;
            if(std::fabs(x_len-y_len)>FLOAT_ZERO || std::fabs(y_len-z_len)>FLOAT_ZERO){
                return false;
            }
            else{
                return true;
            }
        }
        float get_half_length() const{
            if(isCubic()){
                return (max_p.x-min_p.x)*0.86603f;
            }
            else{
//                print("error! not cubic aabb!");
                spdlog::error("error! not cubic aabb!");
                return 0.f;
            }
        }
    public:
        glm::vec3 min_p,max_p;
        std::array<uint32_t ,4> index;//x,y,z and lod
    };

/**
 * obb intersect algorithm:
 * https://blog.csdn.net/silangquan/article/details/50812425?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-5.control&dist_request_id=&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-5.control
 */

    class OBB{
    public:
        OBB(glm::vec3 center_pos,glm::vec3 x,glm::vec3 y,glm::vec3 z,float half_x,float half_y,float half_z)
        :center_pos(center_pos),unit_x(glm::normalize(x)),unit_y(glm::normalize(y)),unit_z(glm::normalize(z)),half_wx(half_x),half_wy(half_y),half_wz(half_z)
        {
            updateHalfLenVec();
        }

        OBB(glm::vec3 center_pos,glm::vec3 x,glm::vec3 y,glm::vec3 z)
        :center_pos(center_pos),unit_x(glm::normalize(x)),unit_y(glm::normalize(y)),unit_z(glm::normalize(z))
        ,hx(x),hy(y),hz(z)
        {
            this->half_wx=glm::length(x)?glm::length(x)/glm::length(unit_x):0;
            this->half_wy=glm::length(y)?glm::length(y)/glm::length(unit_y):0;
            this->half_wz=glm::length(z)?glm::length(z)/glm::length(unit_z):0;
        }

        void updateHalfLenVec(){
            hx=half_wx*unit_x;
            hy=half_wy*unit_y;
            hz=half_wz*unit_z;
        }
        OBB(const OBB& obb){
            this->center_pos=obb.center_pos;
            this->unit_x=obb.unit_x;
            this->unit_y=obb.unit_y;
            this->unit_z=obb.unit_z;
            this->half_wx=obb.half_wx;
            this->half_wy=obb.half_wy;
            this->half_wz=obb.half_wz;
        }
        OBB& operator=(const OBB& obb){
            this->center_pos=obb.center_pos;
            this->unit_x=obb.unit_x;
            this->unit_y=obb.unit_y;
            this->unit_z=obb.unit_z;
            this->half_wx=obb.half_wx;
            this->half_wy=obb.half_wy;
            this->half_wz=obb.half_wz;
            return *this;
        }

        AABB getAABB() const{
            std::vector<glm::vec3> pts;
            for(int i=-1;i<=1;i+=2){
                for(int j=-1;j<=1;j+=2){
                    for(int k=-1;k<=1;k+=2){
                        pts.push_back(center_pos+(float)i*hx+(float)j*hy+(float)k*hz);
                    }
                }
            }
            AABB aabb;
            for(auto& it:pts){
                aabb.Union(it);
            }
            return aabb;
        }

        friend class RayCastCamera;

        bool intersect_aabb(const AABB& aabb)=delete;
        bool intersect_obb(const OBB& obb) const;

    private:
        glm::vec3 center_pos;
        glm::vec3 unit_x;
        glm::vec3 unit_y;
        glm::vec3 unit_z;
        float half_wx;
        float half_wy;
        float half_wz;
        glm::vec3 hx,hy,hz;//normalized direction
    };

    inline bool OBB::intersect_obb(const OBB &obb) const
    {
        auto T=obb.center_pos-this->center_pos;

        float t,ta,tb;

        std::vector<glm::vec3> Ls;
        Ls.push_back(this->unit_x);
        Ls.push_back(this->unit_y);
        Ls.push_back(this->unit_z);
        Ls.push_back(obb.unit_x);
        Ls.push_back(obb.unit_y);
        Ls.push_back(obb.unit_z);
        Ls.push_back(glm::cross(this->unit_x,obb.unit_x));
        Ls.push_back(glm::cross(this->unit_x,obb.unit_y));
        Ls.push_back(glm::cross(this->unit_x,obb.unit_z));
        Ls.push_back(glm::cross(this->unit_y,obb.unit_x));
        Ls.push_back(glm::cross(this->unit_y,obb.unit_y));
        Ls.push_back(glm::cross(this->unit_y,obb.unit_z));
        Ls.push_back(glm::cross(this->unit_z,obb.unit_x));
        Ls.push_back(glm::cross(this->unit_z,obb.unit_y));
        Ls.push_back(glm::cross(this->unit_z,obb.unit_z));

        for(auto& L:Ls){
            t=fabs(glm::dot(T,L));
            ta=fabs(glm::dot(this->hx,L))+fabs(glm::dot(this->hy,L))+fabs(glm::dot(this->hz,L));
            tb=fabs(glm::dot(obb.hx,L))+fabs(glm::dot(obb.hy,L))+fabs(glm::dot(obb.hz,L));
//            print("t: ",t,"ta: ",ta,"tb: ",tb);
            if(t>ta+tb){
//                std::cout<<"not intersect with obb"<<std::endl;
                return false;
            }
        }
        return true;
    }


    inline OBB AABB::convertToOBB() const{
        return OBB((min_p+max_p)/2.f,{1.f,0.f,0.f},{0.f,1.f,0.f},{0.f,0.f,1.f},
                   (max_p.x-min_p.x)/2.f,(max_p.y-min_p.y)/2.f,(max_p.z-min_p.z)/2.f);
    }
    inline OBB AABBToOBB(const AABB& aabb){
        return aabb.convertToOBB();
    }
VS_END





#endif //VOLUMERENDER_BOUNDINGBOX_H
