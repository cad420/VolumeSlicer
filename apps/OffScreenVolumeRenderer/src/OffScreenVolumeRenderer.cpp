 //
// Created by wyz on 2021/9/28.
//
#include <spdlog/sinks/rotating_file_sink.h>
#include <VolumeSlicer/render.hpp>
#include <VolumeSlicer/Utils/timer.hpp>
#include "OffScreenVolumeRenderer.hpp"
#include "json.hpp"
#include <fstream>
#include <filesystem>
using namespace vs;
using nlohmann::json;
static std::shared_ptr<spdlog::logger> logger;
static std::string GetName(const std::string& name){
    auto pos = name.find_last_of('/');
    std::string s = name.substr(pos+1);
    pos = s.find_last_of('.');
    return s.substr(0,pos);
}
static void InitLogger(const std::string& name){
    logger = spdlog::get("offscreen_render_logger");
    if(!logger)
        logger=spdlog::rotating_logger_mt("offscreen_render_logger","logs/"+GetName(name)+"_logfile.txt",1024*1024*10,10);
}
static void LogInfo(std::string str){
    logger->info(str);
}
static void LogError(std::string str){
    logger->error(str);
}

void OffScreenVolumeRenderer::RenderFrames(const char *config_file)
{
    InitLogger(config_file);
    LogInfo("========New Render Start========");
    std::ifstream in;
    in.open(config_file);
    if(!in.is_open()){
        LogError("Can't open config file");
        throw std::runtime_error("Open config file failed");
    }
    json j;
    in >> j;
    in.close();

    int fps;
    if(j.find("fps") != j.end()){
        fps = j.at("fps");
    }
    else{
        fps = 30;
        LogError("Not provide fps, use default fps(30)");
    }

    std::string backend;
    if(j.find("backend") != j.end()){
        backend = j.at("backend");
    }
    else{
        backend = "cpu";
        LogError("Not provide backend, use default backend(cpu)");
    }
    int iGPU;
    if(j.find("iGPU") != j.end()){
        iGPU = j.at("iGPU");
    }
    else{
        iGPU = 0;
        LogError("Not provide iGPU, use default iGPU(0)");
    }
    int width;
    int height;
    if(j.find("width") != j.end() && j.find("height") != j.end()){
        width  = j.at("width");
        height = j.at("height");
    }
    else{
        width  = 1920;
        height = 1080;
        LogError("Not provide width or height, use default width(1920) and height(1080)");
    }
    std::string output_video_name;
    if(j.find("output_video_name") != j.end()){
        output_video_name = j.at("output_video_name");
    }
    else{
        output_video_name = "offscreen-volume-render-video.avi";
        LogError("Not provide output_video_name, use default (offscreen-volume-render-video.avi)");
    }

    bool save_image;
    if(j.find("save_image") != j.end()){
        save_image = j.at("save_image") == "yes";
    }
    else{
        save_image = false;
        LogError("Not provide save_image, use default (no)");
    }
    std::string save_image_path;
    if(j.find("save_image_path") != j.end()){
        save_image_path = j.at("save_image_path");
    }
    else{
        save_image_path = "images";
        LogError("Not provide save_image_path, use default (images)");
    }
    std::string volume_data_config;
    if(j.find("volume_data_config") != j.end()){
        volume_data_config = j.at("volume_data_config");
    }
    else{
        volume_data_config = "mouse_file_config.json";
        LogError("Not provide volume_data_config, use default (mouse_file_config.json)");
    }
    double space_x;
    double space_y;
    double space_z;
    if(j.find("space") != j.end()){
        auto space = j.at("space");
        space_x = space[0];
        space_y = space[1];
        space_z = space[2];
    }
    else{
        space_x = space_y = space_z = 0.001;
        LogError("Not provide space, use default (0.001,0.001,0.001)");
    }

    int max_lod = sizeof(CompRenderPolicy::lod_dist) / sizeof(CompRenderPolicy::lod_dist[0]);
    std::vector<double> lod_policy(max_lod);
    if(j.find("lod_policy") != j.end()){
        auto v = j.at("lod_policy");
        for(int i = 0 ; i < v.size()-1 ; i++){
            lod_policy[i] = v[i];
        }
        if(v.size() <= lod_policy.size() && v.back() == -1.0){
            lod_policy[v.size()-1] = std::numeric_limits<double>::max();
        }
    }
    else{
        lod_policy[0] = std::numeric_limits<double>::max();
        LogError("Not provide lod_policy, use default (lod0 for every)");
    }

    std::unique_ptr<IOffScreenCompVolumeRenderer> renderer;
    SetCUDACtx(iGPU);//used for cuda-renderer and volume
    if(backend=="cuda"){
        renderer=CUDAOffScreenCompVolumeRenderer::Create(width,height);
    }
    else if(backend=="cpu"){
        renderer=CPUOffScreenCompVolumeRenderer::Create(width,height);
    }
    else{
        throw std::runtime_error("Unknown backend, only support cuda or cpu");
    }

    auto volume = CompVolume::Load(volume_data_config.c_str());
    if(space_x!=0.0 && space_y!=0.0 && space_z!=0.0)
    {
        volume->SetSpaceX(space_x);
        volume->SetSpaceY(space_y);
        volume->SetSpaceZ(space_z);
    }
    renderer->SetVolume(std::move(volume));

    TransferFunc tf;
    if(j.find("tf") != j.end()){
        LogInfo("Find user defined transfer function");
        auto points = j.at("tf");
        for(auto it = points.begin();it != points.end(); it++){
            int  key    = std::stoi(it.key());
            auto values = it.value();
            tf.points.emplace_back(key,std::array<double,4>{values[0],values[1],values[2],values[3]});
        }
    }
    else{
        LogError("Not provide transfer function, use default linear gray");
        tf.points.emplace_back(0,std::array<double,4>{0.0,0.0,0.0,0.0});
        tf.points.emplace_back(255,std::array<double,4>{1.0,1.0,1.0,1.0});
    }
    renderer->SetTransferFunc(std::move(tf));

    CompRenderPolicy policy;
    for(int i = 0 ; i < max_lod ; i++){
        policy.lod_dist[i] = lod_policy[i];
    }
    renderer->SetRenderPolicy(policy);

    std::string camera_sequence_config;
    if(j.find("camera_sequence_config") != j.end()){
        camera_sequence_config = j.at("camera_sequence_config");
    }
    else{
        camera_sequence_config = "camera_sequence_config.json";
        LogError("Not provide camera_sequence_config, use default (camera_sequence_config.json)");
    }
    in.open(camera_sequence_config);
    if(!in.is_open()){
        LogError("Open camera sequence config file failed");
        throw std::runtime_error("Open camera sequence config file failed");
    }
    j.clear();
    in >> j;
    int frame_count;
    if(j.find("frame_count") != j.end()){
        frame_count = j.at("frame_count");
    }
    else{
        frame_count = 0;
        LogError("Not provide frame_count, use default frame_count(0)");
    }
    if(j.find("property") != j.end()){
        auto property = j.at("property");
        bool b0 = property[0] == "zoom";
        bool b1 = property[1] == "pos";
        bool b2 = property[2] == "look_at";
        bool b3 = property[3] == "up";
        bool b4 = property[4] == "right";
        if(!b0 || !b1 || !b2 || !b3 || !b4){
            LogError("Camera property order or name not correct");
            throw std::runtime_error("Camera property order or name not correct");
        }
    }
    std::vector<Camera> camera_sequence;
    camera_sequence.reserve(frame_count);
    for(int i=0;i<frame_count;i++){
        auto frame_idx = "frame_"+std::to_string(i+1);
        auto frame_camera = j.at(frame_idx);
        Camera camera{frame_camera[0],
                      {frame_camera[1][0],frame_camera[1][1],frame_camera[1][2]},
                      {frame_camera[2][0],frame_camera[2][1],frame_camera[2][2]},
                      {frame_camera[3][0],frame_camera[3][1],frame_camera[3][2]},
                      {frame_camera[4][0],frame_camera[4][1],frame_camera[4][2]}};
        camera_sequence.push_back(camera);
    }

    VideoCapture video_capture(output_video_name.c_str(),width,height,fps);
    for(int i = 0 ; i < frame_count ; i++){
        Timer timer;
        timer.start();
        renderer->SetCamera(camera_sequence[i]);
        renderer->render(true);
        auto image = renderer->GetImage();
        if(save_image){
            if(!std::filesystem::exists("images")){
                std::filesystem::create_directory("images");
            }
            image.SaveToFile((save_image_path+"/"+GetName(output_video_name)+"_frame_"+std::to_string(i)+".jpeg").c_str());
        }
        std::cout<<"before"<<" "<<i<<std::endl;
        auto img = image.ToImage3b();
        video_capture.AddFrame(reinterpret_cast<uint8_t*>(img.GetData()));
        std::cout<<"after"<<std::endl;
        timer.stop();
        spdlog::set_level(spdlog::level::info);
        LogInfo("render frame "+std::to_string(i)+" cost time "+timer.duration().s().fmt());
        //for gpu take a rest
        _sleep(2000);
    }
    LogInfo("========Last Render Finish========");
}
void OffScreenVolumeRenderer::RenderFrames(OffScreenVolumeRenderer::RenderConfig config,const Callback& callback)
{
    std::unique_ptr<IOffScreenCompVolumeRenderer> renderer;
    SetCUDACtx(config.iGPU);
    if(config.backend=="cuda"){
        renderer=CUDAOffScreenCompVolumeRenderer::Create(config.width,config.height);
    }
    else if(config.backend=="cpu"){
        renderer=CPUOffScreenCompVolumeRenderer::Create(config.width,config.height);
    }
    else{
        throw std::runtime_error("Unknown backend, only support cuda or cpu");
    }

    auto volume = CompVolume::Load(config.volume_data_config_file.c_str());
    if(config.space_x!=0.0 && config.space_y!=0.0 && config.space_z!=0.0)
    {
        volume->SetSpaceX(config.space_x);
        volume->SetSpaceY(config.space_y);
        volume->SetSpaceZ(config.space_z);
    }
    renderer->SetVolume(std::move(volume));

    renderer->SetTransferFunc(std::move(config.tf));

    CompRenderPolicy policy;
    for(int i = 0 ; i < 10 ; i++){
        policy.lod_dist[i] = config.lod_policy[i];
    }
    renderer->SetRenderPolicy(policy);

    nlohmann::json j;

    std::ifstream in;
    in.open(config.camera_sequence_config);
    if(!in.is_open()){
        LogError("Open camera sequence config file failed");
        throw std::runtime_error("Open camera sequence config file failed");
    }
    j.clear();
    in >> j;
    int frame_count;
    if(j.find("frame_count") != j.end()){
        frame_count = j.at("frame_count");
    }
    else{
        frame_count = 0;
        LogError("Not provide frame_count, use default frame_count(0)");
    }
    if(j.find("property") != j.end()){
        auto property = j.at("property");
        bool b0 = property[0] == "zoom";
        bool b1 = property[1] == "pos";
        bool b2 = property[2] == "look_at";
        bool b3 = property[3] == "up";
        bool b4 = property[4] == "right";
        if(!b0 || !b1 || !b2 || !b3 || !b4){
            LogError("Camera property order or name not correct");
            throw std::runtime_error("Camera property order or name not correct");
        }
    }
    std::vector<Camera> camera_sequence;
    camera_sequence.reserve(frame_count);
    for(int i=0;i<frame_count;i++){
        auto frame_idx = "frame_"+std::to_string(i+1);
        auto frame_camera = j.at(frame_idx);
        Camera camera{frame_camera[0],
                      {frame_camera[1][0],frame_camera[1][1],frame_camera[1][2]},
                      {frame_camera[2][0],frame_camera[2][1],frame_camera[2][2]},
                      {frame_camera[3][0],frame_camera[3][1],frame_camera[3][2]},
                      {frame_camera[4][0],frame_camera[4][1],frame_camera[4][2]}};
        camera_sequence.push_back(camera);
    }

    VideoCapture video_capture(config.output_video_name.c_str(),config.width,config.height,config.fps);
    for(int i = 0 ; i < frame_count ; i++){
        Timer timer;
        timer.start();
        renderer->SetCamera(camera_sequence[i]);
        renderer->render(true);
        auto image = renderer->GetImage();
        if(config.save_image){
            if(!std::filesystem::exists("images")){
                std::filesystem::create_directory("images");
            }
            image.SaveToFile((config.image_save_path+"/"+GetName(config.output_video_name)+"_frame_"+std::to_string(i)+".jpeg").c_str());
        }
        std::cout<<"before"<<" "<<i<<std::endl;
        auto img = image.ToImage3b();
        video_capture.AddFrame(reinterpret_cast<uint8_t*>(img.GetData()));
        std::cout<<"after"<<std::endl;
        timer.stop();
        spdlog::set_level(spdlog::level::info);
        LogInfo("render frame "+std::to_string(i)+" cost time "+timer.duration().s().fmt());
        callback(i,(i+1.0)/frame_count,reinterpret_cast<uint8_t*>(image.GetData()));
        //for gpu take a rest
        _sleep(2000);
    }
}
OffScreenVolumeRenderer::RenderConfig OffScreenVolumeRenderer::LoadRenderConfigFromFile(const char *config_file)
{
    InitLogger(config_file);
    std::ifstream in;
    in.open(config_file);
    if(!in.is_open()){
        LogError("Can't open config file");
        throw std::runtime_error("Open config file failed");
    }
    json j;
    in >> j;
    in.close();

    int fps;
    if(j.find("fps") != j.end()){
        fps = j.at("fps");
    }
    else{
        fps = 30;
        LogError("Not provide fps, use default fps(30)");
    }

    std::string backend;
    if(j.find("backend") != j.end()){
        backend = j.at("backend");
    }
    else{
        backend = "cpu";
        LogError("Not provide backend, use default backend(cpu)");
    }
    int iGPU;
    if(j.find("iGPU") != j.end()){
        iGPU = j.at("iGPU");
    }
    else{
        iGPU = 0;
        LogError("Not provide iGPU, use default iGPU(0)");
    }
    int width;
    int height;
    if(j.find("width") != j.end() && j.find("height") != j.end()){
        width  = j.at("width");
        height = j.at("height");
    }
    else{
        width  = 1920;
        height = 1080;
        LogError("Not provide width or height, use default width(1920) and height(1080)");
    }
    std::string output_video_name;
    if(j.find("output_video_name") != j.end()){
        output_video_name = j.at("output_video_name");
    }
    else{
        output_video_name = "offscreen-volume-render-video.avi";
        LogError("Not provide output_video_name, use default (offscreen-volume-render-video.avi)");
    }

    bool save_image;
    if(j.find("save_image") != j.end()){
        save_image = j.at("save_image") == "yes";
    }
    else{
        save_image = false;
        LogError("Not provide save_image, use default (no)");
    }
    std::string save_image_path;
    if(j.find("save_image_path") != j.end()){
        save_image_path = j.at("save_image_path");
    }
    else{
        save_image_path = "images";
        LogError("Not provide save_image_path, use default (images)");
    }
    std::string volume_data_config;
    if(j.find("volume_data_config") != j.end()){
        volume_data_config = j.at("volume_data_config");
    }
    else{
        volume_data_config = "mouse_file_config.json";
        LogError("Not provide volume_data_config, use default (mouse_file_config.json)");
    }
    double space_x;
    double space_y;
    double space_z;
    if(j.find("space") != j.end()){
        auto space = j.at("space");
        space_x = space[0];
        space_y = space[1];
        space_z = space[2];
    }
    else{
        space_x = space_y = space_z = 0.001;
        LogError("Not provide space, use default (0.001,0.001,0.001)");
    }

    int max_lod = sizeof(CompRenderPolicy::lod_dist) / sizeof(CompRenderPolicy::lod_dist[0]);
    std::vector<double> lod_policy(max_lod);
    if(j.find("lod_policy") != j.end()){
        auto v = j.at("lod_policy");
        for(int i = 0 ; i < v.size()-1 ; i++){
            lod_policy[i] = v[i];
        }
        if(v.size() <= lod_policy.size() && v.back() == -1.0){
            lod_policy[v.size()-1] = std::numeric_limits<double>::max();
        }
    }
    else{
        lod_policy[0] = std::numeric_limits<double>::max();
        LogError("Not provide lod_policy, use default (lod0 for every)");
    }

    TransferFunc tf;
    if(j.find("tf") != j.end()){
        LogInfo("Find user defined transfer function");
        auto points = j.at("tf");
        for(auto it = points.begin();it != points.end(); it++){
            int  key    = std::stoi(it.key());
            auto values = it.value();
            tf.points.emplace_back(key,std::array<double,4>{values[0],values[1],values[2],values[3]});
        }
    }
    else{
        LogError("Not provide transfer function, use default linear gray");
        tf.points.emplace_back(0,std::array<double,4>{0.0,0.0,0.0,0.0});
        tf.points.emplace_back(255,std::array<double,4>{1.0,1.0,1.0,1.0});
    }

    std::string camera_sequence_config;
    if(j.find("camera_sequence_config") != j.end()){
        camera_sequence_config = j.at("camera_sequence_config");
    }
    else{
        camera_sequence_config = "camera_sequence_config.json";
        LogError("Not provide camera_sequence_config, use default (camera_sequence_config.json)");
    }

    RenderConfig render_config;
    render_config.fps = fps;
    render_config.backend = backend;
    render_config.iGPU = iGPU;
    render_config.width = width;
    render_config.height = height;
    render_config.output_video_name = output_video_name;
    render_config.save_image = save_image;
    render_config.volume_data_config_file = volume_data_config;
    render_config.space_x = space_x;
    render_config.space_y = space_y;
    render_config.space_z = space_z;
    memcpy(render_config.lod_policy,lod_policy.data(),lod_policy.size()*sizeof(double));
    render_config.tf = tf;
    render_config.camera_sequence_config = camera_sequence_config;
    render_config.image_save_path = save_image_path;
    return render_config;
}
auto OffScreenVolumeRenderer::LoadCameraSequenceFromFile(const char *camera_file)->std::vector<Camera>
{
    nlohmann::json j;

    std::ifstream in;
    in.open(camera_file);
    if(!in.is_open()){
        LogError("Open camera sequence config file failed");
        throw std::runtime_error("Open camera sequence config file failed");
    }
    j.clear();
    in >> j;
    int frame_count;
    if(j.find("frame_count") != j.end()){
        frame_count = j.at("frame_count");
    }
    else{
        frame_count = 0;
        LogError("Not provide frame_count, use default frame_count(0)");
    }
    if(j.find("property") != j.end()){
        auto property = j.at("property");
        bool b0 = property[0] == "zoom";
        bool b1 = property[1] == "pos";
        bool b2 = property[2] == "look_at";
        bool b3 = property[3] == "up";
        bool b4 = property[4] == "right";
        if(!b0 || !b1 || !b2 || !b3 || !b4){
            LogError("Camera property order or name not correct");
            throw std::runtime_error("Camera property order or name not correct");
        }
    }
    std::vector<Camera> camera_sequence;
    camera_sequence.reserve(frame_count);
    for(int i=0;i<frame_count;i++){
        auto frame_idx = "frame_"+std::to_string(i+1);
        auto frame_camera = j.at(frame_idx);
        Camera camera{frame_camera[0],
                      {frame_camera[1][0],frame_camera[1][1],frame_camera[1][2]},
                      {frame_camera[2][0],frame_camera[2][1],frame_camera[2][2]},
                      {frame_camera[3][0],frame_camera[3][1],frame_camera[3][2]},
                      {frame_camera[4][0],frame_camera[4][1],frame_camera[4][2]}};
        camera_sequence.push_back(camera);
    }

    return camera_sequence;
}
