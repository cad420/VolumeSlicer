//
// Created by wyz on 2021/5/29.
//
#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <iostream>
namespace control{

    class TrackBall{
    public:
        enum Mode:int{
            ARC=0,
            PROJECT=1
        };
        TrackBall():radius(0),screen_height(0),screen_width(0),half_screen_w(0.f),half_screen_h(0.f),mode(ARC){}
        TrackBall(float radius,int screen_w,int screen_h):radius(radius),screen_width(screen_w),screen_height(screen_h),
        half_screen_w(0.5f*screen_w),half_screen_h(0.5f*screen_h),mode(ARC){}
        virtual ~TrackBall(){}

        void printSelf() const{
            std::cout<<"=====TrackBall Info=====\n"
                     <<"     Radius: "<<radius<<"\n"
                     <<"Screen Size: "<<screen_width<<" "<<screen_height<<"\n"
                     <<"     Mode: "<<(mode==ARC?"ARC\n":"PROJECT\n")
                     <<std::endl;
        }

        void set(float r,int w,int h){
            this->radius=r;
            this->screen_width=w;
            this->screen_height=h;
            this->half_screen_w=w*0.5f;
            this->half_screen_h=h*0.5f;
        }
        void setScreenSize(int w,int h){
            this->screen_width=w;
            this->screen_height=h;
            this->half_screen_w=w*0.5f;
            this->half_screen_h=h*0.5f;
        }
        void setRadius(float r) { this->radius=r; }
        void setMode(TrackBall::Mode mode) {this->mode=mode;}

        float getRadius() const { return this->radius; }
        int getScreenW() const { return this->screen_width; }
        int getScreenH() const { return this->screen_height; }
        TrackBall::Mode getMode() const { return this->mode; }
        glm::vec3 getVector(int x,int y) const{
            if(radius==0.f || !screen_width || !screen_height){
                return {0.f,0.f,0.f};
            }
            float mx=x-half_screen_w;
            float my=half_screen_h-y;
            return mode==ARC?getVectorWithArc(mx,my):getVectorWithProject(mx,my);
        }
        glm::vec3 getUnitVector(int x,int y) const{
            return glm::normalize(this->getVector(x,y));
        }

    private:
        glm::vec3 getVectorWithArc(float x,float y) const;
        glm::vec3 getVectorWithProject(float x,float y) const;
        float clampX(float x) const{
            if(x<=-half_screen_w)
                x=-half_screen_w+1;
            else if(x>=half_screen_w)
                x=half_screen_w-1;
            return x;
        }
        float clampY(float y) const{
            if(y<=-half_screen_h)
                y=-half_screen_h+1;
            else if(y>=half_screen_h)
                y=half_screen_h-1;
            return y;
        }
    private:
        float radius;
        int screen_width;
        int screen_height;
        float half_screen_w;
        float half_screen_h;
        TrackBall::Mode mode;
    };

    inline glm::vec3 TrackBall::getVectorWithArc(float x, float y) const {
        x*=0.01f;
        y*=0.01f;
        float arc=sqrtf(x*x+y*y);
        float a=arc/radius;
        float b=atan2f(y,x);
        float x2=radius*sinf(a);

        glm::vec3 pos_on_ball;
        pos_on_ball.x=x2*cosf(b);
        pos_on_ball.y=x2*sinf(b);
        pos_on_ball.z=radius*cosf(a);
        return pos_on_ball;

    }

    inline glm::vec3 TrackBall::getVectorWithProject(float x, float y) const {
        return glm::vec3();
    }

    enum class CameraDefinedKey{
        Forward,
        Backward,
        Left,
        Right,
        Up,
        Bottom
    };
    enum class CameraDefinedMouseButton{
        Left,Right,Middle
    };
    class Camera{
    public:
        Camera()=default;
        virtual ~Camera(){}
        virtual void processMouseScroll(float yoffset)=0;
        virtual void processMouseMove(double x_pos,double y_pos)=0;
        virtual void processMouseButton(CameraDefinedMouseButton button,bool press,double x_pos,double y_pos)=0;
        virtual void processKeyEvent(CameraDefinedKey key,float t)=0;
        virtual glm::mat4 getViewMatrix()=0;
        virtual float getZoom() const =0;
        virtual glm::vec3 getCameraPos() const =0;
        virtual glm::vec3 getCameraLookAt() const =0;
        virtual glm::vec3 getCameraUp() const =0;
    };
    class TrackBallCamera: public TrackBall, public Camera{
    public:
        TrackBallCamera(float radius,int screen_w,int screen_h,glm::vec3 ball_center_pos)
        :TrackBall(radius,screen_w,screen_h),ball_center_pos(ball_center_pos)
        {
            cur_quat=pre_quat={1.f,0.f,0.f,0.f};
            eye_pos=glm::vec3(ball_center_pos)+glm::vec3(0.f,0.f,radius*2);
            look_at=ball_center_pos;
            up={0.f,1.f,0.f};

        }
        void processMouseScroll(float yoffset) override;
        void processMouseMove(double x_pos,double y_pos) override;
        void processMouseButton(CameraDefinedMouseButton button,bool press,double x_pos,double y_pos) override;
        void processKeyEvent(CameraDefinedKey key,float t) override{};
        glm::mat4 getViewMatrix() override;
        float getZoom() const override;
        glm::vec3 getCameraPos() const override;
        glm::vec3 getCameraLookAt() const override;
        glm::vec3 getCameraUp() const override;
    private:
        glm::qua<float> cur_quat;
        glm::qua<float> pre_quat;
        glm::vec3 eye_pos,look_at,up;
        double pre_mouse_x,pre_mouse_y;
        glm::vec3 ball_center_pos;
    };

    inline void TrackBallCamera::processMouseScroll(float yoffset) {
        if(yoffset>0.f){
            eye_pos+=glm::vec3(0.f,0.f,0.1f);

        }
        else{
            eye_pos-=glm::vec3(0.f,0.f,0.1f);
            if(eye_pos.z<ball_center_pos.z)
                eye_pos.z=ball_center_pos.z+0.01f;
        }
//        std::cout<<eye_pos.x<<" "<<eye_pos.y<<" "<<eye_pos.z<<std::endl;
    }

    inline void TrackBallCamera::processMouseButton(CameraDefinedMouseButton button, bool press, double x_pos, double y_pos) {
        if(button==CameraDefinedMouseButton::Left && press){
            pre_mouse_x=x_pos;
            pre_mouse_y=y_pos;
            pre_quat=cur_quat;
        }
    }

    inline void TrackBallCamera::processMouseMove(double x_pos, double y_pos) {
        auto v1=getUnitVector(pre_mouse_x,pre_mouse_y);
        auto v2=getUnitVector(x_pos,y_pos);
        auto delta=glm::qua<float>(v1,v2);
        cur_quat=delta*pre_quat;

    }

    inline float TrackBallCamera::getZoom() const {
        return 45.f;
    }

    inline glm::vec3 TrackBallCamera::getCameraPos() const {

        auto m=glm::translate(glm::mat4(1.f),-ball_center_pos);

        auto mm=glm::translate(glm::mat4(1.f),ball_center_pos);

        return mm*glm::transpose(glm::mat4_cast(cur_quat))*m*glm::vec4(eye_pos,1.f);
    }

    inline glm::mat4 TrackBallCamera::getViewMatrix() {
        auto m=glm::translate(glm::mat4(1.f),-ball_center_pos);

        auto mm=glm::translate(glm::mat4(1.f),ball_center_pos);

        return glm::lookAt(eye_pos,look_at,up)*mm*glm::mat4_cast(cur_quat)*m;
    }

    inline glm::vec3 TrackBallCamera::getCameraLookAt() const {
        auto m=glm::translate(glm::mat4(1.f),-ball_center_pos);

        auto mm=glm::translate(glm::mat4(1.f),ball_center_pos);

        return mm*glm::transpose(glm::mat4_cast(cur_quat))*m*glm::vec4(look_at,1.f);
    }

    inline glm::vec3 TrackBallCamera::getCameraUp() const {
        auto m=glm::translate(glm::mat4(1.f),-ball_center_pos);

        auto mm=glm::translate(glm::mat4(1.f),ball_center_pos);

        return mm*glm::transpose(glm::mat4_cast(cur_quat))*m*glm::vec4(up,0.f);
    }


    const float PITCH=0.0f;
    const float YAW=-90.0f;
    const float SPEED=20.f;
    const float SENSITIVITY=0.3f;
    const float ZOOM=20.0f;
    class FPSCamera: public Camera{
    public:
        FPSCamera(glm::vec3 pos)
        :pos(pos){
            world_up={0.f,1.f,0.f};

            yaw=YAW;
            pitch=PITCH;
            move_speed=SPEED;
            move_sense=SENSITIVITY;
            zoom=ZOOM;

            space_ratio={1.f,1.f,1.f};
            updateVector();
        }
        void setSpaceRatio(float x, float y, float z){
            space_ratio={x,y,z};
        }
        void processMouseScroll(float yoffset) override;
        void processMouseMove(double x_pos,double y_pos) override;
        void processMouseButton(CameraDefinedMouseButton button,bool press,double x_pos,double y_pos)override{
            if(button==CameraDefinedMouseButton::Left && press){
                last_x=x_pos;
                last_y=y_pos;
            }
        };
        void processKeyEvent(CameraDefinedKey key,float t) override;
        glm::mat4 getViewMatrix() override;
        glm::vec3 getCameraPos() const override{
            return pos;
        }
        float getZoom()const override{
            return zoom;
        }
        glm::vec3 getCameraLookAt() const override{
            return view_direction+pos;
        }
        glm::vec3 getCameraUp() const override{
            return up;
        }
        glm::vec3 getCameraRight() const{
            return right;
        }
    private:
        void updateVector();
    private:
        glm::vec3 pos,view_direction,up,right,world_up;
        float yaw,pitch,move_speed,move_sense,zoom;
        glm::vec3 space_ratio;
        double last_x,last_y;
//        bool first=true;
    };

    inline void FPSCamera::updateVector() {
        glm::vec3 front;
        front.x=cos(glm::radians(yaw))*cos(glm::radians(pitch));
        front.y=sin(glm::radians(pitch));
        front.z=sin(glm::radians(yaw))*cos(glm::radians(pitch));
        view_direction=glm::normalize(front);
        right=glm::normalize(glm::cross(view_direction,world_up));
        up=glm::normalize(glm::cross(right,view_direction));
    }

    inline void FPSCamera::processMouseScroll(float yoffset) {
        zoom-=yoffset;
        if(zoom<0.1f)
            zoom=0.1f;
        if(zoom>45.0f)
            zoom=45.0f;
    }

    inline void FPSCamera::processMouseMove(double x_pos, double y_pos) {

//        if(first){
//            last_x=x_pos;
//            last_y=y_pos;
//            first=false;
//        }
        auto xoffset=x_pos-last_x;
        auto yoffset=last_y-y_pos;
        last_x=x_pos;
        last_y=y_pos;

        xoffset*=move_sense;
        yoffset*=move_sense;

        yaw+=xoffset;
        pitch+=yoffset;


        if (pitch > 60.0f)
            pitch = 60.0f;
        if(pitch<-60.0f)
            pitch=-60.0f;

        updateVector();
    }

    inline void FPSCamera::processKeyEvent(CameraDefinedKey key, float t) {
            float ds=move_speed*t;
            if(key==CameraDefinedKey::Forward)
                pos+=space_ratio*view_direction*ds;
            if(key==CameraDefinedKey::Backward)
                pos-=space_ratio*view_direction*ds;
            if(key==CameraDefinedKey::Left)
                pos-=space_ratio*right*ds;
            if(key==CameraDefinedKey::Right)
                pos+=space_ratio*right*ds;
            if(key==CameraDefinedKey::Up)
                pos+=space_ratio*up*ds;
            if(key==CameraDefinedKey::Bottom)
                pos-=space_ratio*up*ds;

    }

    inline glm::mat4 FPSCamera::getViewMatrix() {
        return glm::lookAt(pos,view_direction+pos,up);
    }

    class TrackingCamera: public Camera{
    public:

    };
}

