
#include "VideoCapture.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
class VideoCaptureImpl{
  public:
    VideoCaptureImpl(const char* filename,int width,int height,int fps);
    void AddFrame(const uint8_t* pixels);
    ~VideoCaptureImpl();
  private:
    int width,height;
    cv::VideoWriter output_video;
};
VideoCaptureImpl::VideoCaptureImpl(const char *filename, int width, int height, int fps)
:width(width),height(height)
{
    output_video.open(filename,cv::VideoWriter::fourcc('M','J','P','G'),fps,{width,height},true);
    if(!output_video.isOpened()){
        throw std::runtime_error("Could not open the output video for write");
    }
}
void VideoCaptureImpl::AddFrame(const uint8_t *pixels)
{
    cv::Mat image(height,width,CV_8UC3,const_cast<uint8_t*>(pixels));
    output_video.write(image);
}
VideoCaptureImpl::~VideoCaptureImpl()
{
    output_video.release();
}
//-----------------------------------
VideoCapture::VideoCapture(const char *filename, int width, int height, int fps)
{
    impl=std::make_unique<VideoCaptureImpl>(filename,width,height,fps);
}
void VideoCapture::AddFrame(const uint8_t *pixels)
{
    impl->AddFrame(pixels);
}
VideoCapture::~VideoCapture()
{
    impl.reset();
}
