//
// Created by wyz on 2021/6/28.
//
#include "SliceZoomWidget.hpp"
#include <QPaintEvent>
#include <QPainter>
#include <VolumeSlicer/frame.hpp>
#include <iostream>
#include <QImage>
#include <glm/glm.hpp>
SliceZoomWidget::SliceZoomWidget(QWidget *parent) {
    initSlicer();
}

void SliceZoomWidget::paintEvent(QPaintEvent *event) {
    if(!slicer || !max_zoom_slicer || !raw_volume || ! raw_volume_sampler) return;
    QPainter p(this);
    Frame frame;
    frame.width=max_zoom_slicer->GetImageW();
    frame.height=max_zoom_slicer->GetImageH();
    frame.channels=1;
    frame.data.resize((size_t)frame.width*frame.height*frame.channels,0);
    raw_volume_sampler->Sample(max_zoom_slicer->GetSlice(),frame.data.data());

    QImage image(frame.data.data(),frame.width,frame.height,QImage::Format::Format_Grayscale8,nullptr,nullptr);
    QImage color_img=image.convertToFormat(QImage::Format_RGBA8888);
    drawSliceLine(color_img);
    p.drawPixmap(0,0,QPixmap::fromImage(color_img.mirrored(false,true)));

    event->accept();
}

void SliceZoomWidget::mouseMoveEvent(QMouseEvent *event) {
    event->accept();
}

void SliceZoomWidget::wheelEvent(QWheelEvent *event) {
    event->accept();
}

void SliceZoomWidget::mousePressEvent(QMouseEvent *event) {
    event->accept();
}

void SliceZoomWidget::mouseReleaseEvent(QMouseEvent *event) {
    event->accept();
}

void SliceZoomWidget::setSlicer(const std::shared_ptr<Slicer> &slicer) {
    if(!slicer) return;

    this->slicer=slicer;
    auto slice=this->slicer->GetSlice();
    slice.origin={slice.origin[0]/64,slice.origin[1]/64,slice.origin[2]/64};
    {
        //planer equation: A(x-x0)+B(y-y0)+C(z-z0)=0
        //                 Ax+By+Cz-(A*x0+B*y0+C*z0)=0
        //                 A(t*x1+(1-t)*x2)+B(t*y1+(1-t)*y2)+C(t*z1+(1-t)*z2)=(A*x0+B*y0+C*z0)
        //                 A*x2+A*(x1-x2)*t+B*y2+B*(y1-y2)*t+C*z2+C*(z1-z2)*t=(A*x0+B*y0+C*z0)
        //                 D=(A*x0+B*y0+C*z0)
        //                 A*x2+A*(x1-x2)*t+B*y2+B*(y1-y2)*t+C*z2+C*(z1-z2)*t = D
        //                 A*x2+B*y2+C*z2+(A*(x1-x2)+B*(y1-y2)+C*(z1-z2))*t   = D
        //                 t = ( D - (A*x2+B*y2+C*z2) ) / (A*(x1-x2)+B*(y1-y2)+C*(z1-z2))
        //                 0 <= t <= 1
        static float space_x=this->raw_volume->GetVolumeSpaceX();
        static float space_y=this->raw_volume->GetVolumeSpaceY();
        static float space_z=this->raw_volume->GetVolumeSpaceZ();
        static float base_space=std::min({space_x,space_y,space_z});
        static float space_ratio_x=space_x/base_space;
        static float space_ratio_y=space_y/base_space;
        static float space_ratio_z=space_z/base_space;
        float A=slice.normal[0]*space_ratio_x;
        float B=slice.normal[1]*space_ratio_y;
        float C=slice.normal[2]*space_ratio_z;
        float length=std::sqrt(A*A+B*B+C*C);
        A/=length;
        B/=length;
        C/=length;
        float x0=slice.origin[0];
        float y0=slice.origin[1];
        float z0=slice.origin[2];
        float D=A*x0+B*y0+C*z0;

        static std::array< std::array<float,3> ,8> pts={
                std::array<float,3>{0.f,0.f,0.f},
                std::array<float,3>{366.f,0.f,0.f},
                std::array<float,3>{366.f,463.f,0.f},
                std::array<float,3>{0.f,463.f,0.f},
                std::array<float,3>{0.f,0.f,161.f},
                std::array<float,3>{366.f,0.f,161.f},
                std::array<float,3>{366.f,463.f,161.f},
                std::array<float,3>{0.f,463.f,161.f}
        };
        //total 12 lines
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
                if(t>=0.f && t<=1.f){
                    intersect_pts_cnt++;
                    tmp={t*x1+(1-t)*x2,t*y1+(1-t)*y2,t*z1+(1-t)*z2};
                    intersect_pts={intersect_pts[0]+tmp[0],
                                   intersect_pts[1]+tmp[1],
                                   intersect_pts[2]+tmp[2]};
                }
            }
        }
//        std::cout<<"intersect pts cnt: "<<intersect_pts_cnt<<std::endl;
        intersect_pts={intersect_pts[0]/intersect_pts_cnt,
                       intersect_pts[1]/intersect_pts_cnt,
                       intersect_pts[2]/intersect_pts_cnt};
//        std::cout<<"intersect pt pos: "<<intersect_pts[0]<<" "
//                                       <<intersect_pts[1]<<" "
//                                       <<intersect_pts[2]<<std::endl;
        slice.origin={intersect_pts[0],
                      intersect_pts[1],
                      intersect_pts[2],1.f};
    }
    std::cout<<slice.origin[0]<<" "<<slice.origin[1]<<" "<< slice.origin[2]<<std::endl;
    slice.voxel_per_pixel_width=1.3;
    slice.voxel_per_pixel_height=1.3;
    slice.n_pixels_height=400;
    slice.n_pixels_width=400;
    this->max_zoom_slicer=Slicer::CreateSlicer(slice);
}
void SliceZoomWidget::drawSliceLine( QImage& image) {

    static float space_x=this->raw_volume->GetVolumeSpaceX();
    static float space_y=this->raw_volume->GetVolumeSpaceY();
    static float space_z=this->raw_volume->GetVolumeSpaceZ();
    static float base_space=std::min({space_x,space_y,space_z});
    static float space_ratio_x=space_x/base_space;
    static float space_ratio_y=space_y/base_space;
    static float space_ratio_z=space_z/base_space;

    auto slice=slicer->GetSlice();
    assert(slice.voxel_per_pixel_height==slice.voxel_per_pixel_width);
    float p=slice.voxel_per_pixel_height/64.f;
//    std::cout<<image.width()<<" "<<image.height()<<std::endl;
    auto max_zoom_slice=max_zoom_slicer->GetSlice();
    glm::vec3 right={slice.right[0],slice.right[1],slice.right[2]};
    glm::vec3 up={slice.up[0],slice.up[1],slice.up[2]};
//    float x_t=std::abs(glm::dot(right,{1,1,3}));
//    float y_t=std::abs(glm::dot(up,{1,1,3}));
    glm::vec3 offset={(slice.origin[0]/64.f-max_zoom_slice.origin[0])*space_ratio_x,
                      (slice.origin[1]/64.f-max_zoom_slice.origin[1])*space_ratio_y,
                      (slice.origin[2]/64.f-max_zoom_slice.origin[2])*space_ratio_z};

    float x_offset=glm::dot(right,offset);
    float y_offset=-glm::dot(up,offset);
    int min_p_x=x_offset/max_zoom_slice.voxel_per_pixel_width
            + max_zoom_slice.n_pixels_width/2 - slice.n_pixels_width/2*p/max_zoom_slice.voxel_per_pixel_width;
    int min_p_y=y_offset/max_zoom_slice.voxel_per_pixel_height
            + max_zoom_slice.n_pixels_height/2 - slice.n_pixels_height/2*p/max_zoom_slice.voxel_per_pixel_width;
    int max_p_x=x_offset/max_zoom_slice.voxel_per_pixel_width
            + max_zoom_slice.n_pixels_width/2 + slice.n_pixels_width/2*p/max_zoom_slice.voxel_per_pixel_width;
    int max_p_y=y_offset/max_zoom_slice.voxel_per_pixel_height
            + max_zoom_slice.n_pixels_height/2 + slice.n_pixels_height/2*p/max_zoom_slice.voxel_per_pixel_width;
    spdlog::info("min_x:{0}, min_y:{1}, max_x:{2}, max_y{3}.",min_p_x,min_p_y,max_p_x,max_p_y);
    min_p_x=min_p_x<0?0:min_p_x;
    max_p_x=max_p_x<max_zoom_slice.n_pixels_width?max_p_x:max_zoom_slice.n_pixels_width-1;
    min_p_y=min_p_y<0?0:min_p_y;
    max_p_y=max_p_y<max_zoom_slice.n_pixels_height?max_p_y:max_zoom_slice.n_pixels_height-1;
    for(auto i=min_p_x;i<=max_p_x;i++){
        image.setPixelColor(i,min_p_y,QColor(255,0,0,255));
        image.setPixelColor(i,max_p_y,QColor(255,0,0,255));
    }
    for(auto i=min_p_y;i<=max_p_y;i++){
        image.setPixelColor(min_p_x,i,QColor(255,0,0,255));
        image.setPixelColor(max_p_x,i,QColor(255,0,0,255));
    }
}
void SliceZoomWidget::redraw() {
    setSlicer(this->slicer);
    repaint();
}

void SliceZoomWidget::initSlicer() {


}

void SliceZoomWidget::setRawVolume(const std::shared_ptr<RawVolume>& raw_volume) {
    this->raw_volume_sampler=VolumeSampler::CreateVolumeSampler(raw_volume);
    this->raw_volume=raw_volume;
}

void SliceZoomWidget::resizeEvent(QResizeEvent *event) {
    QWidget::resizeEvent(event);
}

void SliceZoomWidget::volumeLoaded() {

}

void SliceZoomWidget::volumeClose() {
    spdlog::info("{0}.",__FUNCTION__ );
    slicer.reset();
    max_zoom_slicer.reset();
    raw_volume_sampler.reset();
    raw_volume.reset();
    repaint();
}


