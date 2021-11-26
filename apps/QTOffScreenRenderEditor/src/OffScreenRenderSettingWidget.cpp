#include "OffScreenRenderSettingWidget.hpp"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include "CameraVisWidget.hpp"
#include <QListWidget>
#include <QGroupBox>
#include <QSlider>
#include <QComboBox>
#include <QProcess>
#include <VolumeSlicer/Utils/logger.hpp>
#include <json.hpp>
#include "BSplineCurve.h"
struct RenderConfig;
static auto LoadCameraSequenceFromFile(const std::string&)->std::vector<Camera>;
static void SaveCameraSequenceToFile(const std::string&,const std::vector<Camera>& cameras);
static void SaveRenderConfigToFile(const std::string&,const RenderConfig&);

OffScreenRenderSettingWidget::OffScreenRenderSettingWidget(QWidget *parent)
:QWidget(parent)
{
    auto layout = new QVBoxLayout();
    this->setLayout(layout);

    //Camera Sequence
    auto camera_gb = new QGroupBox("Camera Sequence");
    camera_gb->setFixedSize(480,500);
    auto camera_gb_layout =new QVBoxLayout();
    camera_gb->setLayout(camera_gb_layout);

    auto camera_layout = new QHBoxLayout();
    camera_item_widget = new QListWidget();
    camera_item_widget->setFixedSize(300,120);

    auto camera_pb_layout = new QVBoxLayout();
    camera_load_pb = new QPushButton("Import");
    camera_load_pb->setFixedWidth(60);
    camera_export_pb = new QPushButton("Export");
    camera_export_pb->setFixedWidth(60);
    camera_del_pb = new QPushButton("Delete");
    camera_del_pb->setFixedWidth(60);
    smooth_camera_pb = new QPushButton("Smooth");
    smooth_camera_pb->setFixedWidth(60);
    camera_pb_layout->addWidget(camera_load_pb);
    camera_pb_layout->addWidget(camera_export_pb);
    camera_pb_layout->addWidget(camera_del_pb);
    camera_pb_layout->addWidget(smooth_camera_pb);
    //enable until volume loaded
    camera_load_pb->setEnabled(false);
    camera_export_pb->setEnabled(false);
    camera_del_pb->setEnabled(false);
    smooth_camera_pb->setEnabled(false);

    camera_layout->addWidget(camera_item_widget);
    camera_layout->addLayout(camera_pb_layout);

    camera_gb_layout->addLayout(camera_layout);


    camera_vis_widget = new CameraVisWidget(this);
    camera_vis_widget->setFixedSize(400,300);
    camera_gb_layout->addWidget(camera_vis_widget,0,Qt::AlignmentFlag::AlignCenter);

    layout->addWidget(camera_gb);

    //Off-Screen render config
    auto render_config_gb = new QGroupBox("Off-Screen Render Config");
    render_config_gb->setFixedSize(480,200);
    auto render_layout = new QVBoxLayout();
    render_config_gb->setLayout(render_layout);
    layout->addWidget(render_config_gb);

    auto video_layout = new QHBoxLayout();
    output_video_name_pb = new QPushButton("Video Name As");
    output_video_name_le = new QLineEdit();
    output_video_name_le->setReadOnly(true);
    video_layout->addWidget(output_video_name_pb);
    video_layout->addWidget(output_video_name_le);
    render_layout->addLayout(video_layout);

    auto render_paras_layout = new QHBoxLayout();
    auto render_width_lb = new QLabel("Width");
    auto render_height_lb = new QLabel("Height");
    auto render_fps_lb = new QLabel("FPS");
    render_width_sb = new QSpinBox();
    render_height_sb = new QSpinBox();
    render_width_sb->setRange(1,10000);
    render_height_sb->setRange(1,10000);
    render_fps_sb = new QSpinBox();
    render_fps_sb->setRange(1,60);
    render_paras_layout->addWidget(render_width_lb);
    render_paras_layout->addWidget(render_width_sb);
    render_paras_layout->addWidget(render_height_lb);
    render_paras_layout->addWidget(render_height_sb);
    render_paras_layout->addWidget(render_fps_lb);
    render_paras_layout->addWidget(render_fps_sb);
    render_layout->addLayout(render_paras_layout);

    auto render_policy_layout = new QHBoxLayout();
    auto render_policy_lb = new QLabel("Render Quality");
    render_policy_slider = new QSlider(Qt::Orientation::Horizontal);
    render_policy_slider->setRange(1,5);
    render_policy_layout->addWidget(render_policy_lb);
    render_policy_layout->addWidget(new QLabel("Low"));
    render_policy_layout->addWidget(render_policy_slider);
    render_policy_layout->addWidget(new QLabel("High"));
    render_layout->addLayout(render_policy_layout);

    auto render_cameras_layout = new QHBoxLayout();
    auto render_cameras_lb = new QLabel("Camera Config");
    render_cameras_cb = new QComboBox();
    render_cameras_layout->addWidget(render_cameras_lb);
    render_cameras_layout->addWidget(render_cameras_cb);
    render_layout->addLayout(render_cameras_layout);

    auto render_pb_layout = new QHBoxLayout();
    save_render_config_pb = new QPushButton("Export Render Config");
    start_off_render_pb = new QPushButton("Start Render Program");
    render_pb_layout->addWidget(save_render_config_pb);
    render_pb_layout->addWidget(start_off_render_pb);
    render_layout->addLayout(render_pb_layout);

    connect(camera_load_pb,&QPushButton::clicked,this,[this](){
        this->importCamerasFromFile(
            QFileDialog::getOpenFileName(this,
                                         QStringLiteral("Load Cameras"),
                                         QStringLiteral("."),
                                         QStringLiteral("config files(*.json)")).toStdString()
            );
    });
    connect(camera_export_pb,&QPushButton::clicked,this,[this](){
       if(camera_item_widget->count()==0) return;
       auto items = camera_item_widget->selectedItems();
       if(items.size() != 1) return;
       auto name = items.front()->text().toStdString();
       LOG_INFO("export name {}",name);
       exportCamerasToFile(name);
    });
    connect(camera_del_pb,&QPushButton::clicked,this,[this](){
      if(camera_item_widget->count()==0) return;
      auto items = camera_item_widget->selectedItems();
      for(auto item:items){
          auto name = item->text().toStdString();
          delete camera_item_widget->takeItem(camera_item_widget->row(item));
          deleteCamerasItem(name);
          render_cameras_cb->removeItem(render_cameras_cb->findText(name.c_str()));
      }
    });
    connect(camera_item_widget,&QListWidget::currentTextChanged,this,[this](const QString& name){
        const auto& cameras = camera_map[name.toStdString()];
        sendCameraPosToVis(cameras);
    });
    connect(smooth_camera_pb,&QPushButton::clicked,this,[this](){
        if(camera_item_widget->count()==0) return;
        auto items = camera_item_widget->selectedItems();
        for(auto item:items){
            auto name = item->text().toStdString();
            smoothCamerasItem(name);
        }
    });
    connect(save_render_config_pb,&QPushButton::clicked,this,[this](){
      auto path = QFileDialog::getSaveFileName(
            this,QString("Save As"),
                        QString("."),QString("(*.json)")
            ).toStdString();
      if(path.empty()) return;
      saveOffScreenRenderSettingToFile(path);
    });
    connect(start_off_render_pb,&QPushButton::clicked,this,[this](){
        startRenderProgram();
    });
    connect(output_video_name_pb,&QPushButton::clicked,this,[this](){
      auto path = QFileDialog::getSaveFileName(
          this,QString("Save As"),
          QString("."),QString("(*.avi)")
      );
      if(path.isEmpty()) return;
      output_video_name_le->setText(path);
    });
}

void OffScreenRenderSettingWidget::receiveRecordCameras(std::vector<Camera> cameras)
{
    sendCameraPosToVis(cameras);
    auto name = "camera_"+std::to_string(count++);
    camera_map[name] = std::move(cameras);
    auto item = new QListWidgetItem(name.c_str());
    this->camera_item_widget->addItem(item);
    this->camera_item_widget->setCurrentItem(item);
    this->render_cameras_cb->addItem(name.c_str());
}
void OffScreenRenderSettingWidget::volumeLoaded(const std::shared_ptr<CompVolume> &comp_volume)
{
    {
        auto volume_space_x = comp_volume->GetVolumeSpaceX();
        auto volume_space_y = comp_volume->GetVolumeSpaceY();
        auto volume_space_z = comp_volume->GetVolumeSpaceZ();
        auto base_space = std::sqrt(volume_space_x*volume_space_x+volume_space_y*volume_space_y+volume_space_z*volume_space_z);
        auto block_length = comp_volume->GetBlockLength()[0];
        auto base_lod = base_space * block_length * 2;
        LOG_INFO("base lod {0}",base_lod);
        for(int i=0;i<9;i++){
            low_render_policy[i] = base_lod * std::pow(2,i);
        }
        low_render_policy[9] = std::numeric_limits<double>::max();
    }

    this->camera_vis_widget->volumeLoaded(comp_volume);

    camera_load_pb->setEnabled(true);
    camera_export_pb->setEnabled(true);
    camera_del_pb->setEnabled(true);
    smooth_camera_pb->setEnabled(true);
}
void OffScreenRenderSettingWidget::volumeClosed()
{
    camera_load_pb->setEnabled(false);
    camera_export_pb->setEnabled(false);
    camera_del_pb->setEnabled(false);
    smooth_camera_pb->setEnabled(false);
    this->camera_vis_widget->volumeClosed();
    clear();
}
void OffScreenRenderSettingWidget::clear()
{
    count = 0;
    camera_map.clear();
    camera_item_widget->clear();
}
void OffScreenRenderSettingWidget::importCamerasFromFile(const std::string &path)
{
    try{
        auto import_cameras = LoadCameraSequenceFromFile(path.c_str());
        receiveRecordCameras(std::move(import_cameras));
    }
    catch (const std::exception& err)
    {
        LOG_ERROR("importCamerasFromFile error: {}",err.what());
    }
}

void OffScreenRenderSettingWidget::exportCamerasToFile(const std::string &name)
{
    try{
        const auto& cameras = camera_map.at(name);
        SaveCameraSequenceToFile(name.c_str(),cameras);
    }
    catch (const std::exception& err)
    {
        LOG_ERROR("exportCamerasToFile error: {}",err.what());
    }
}
void OffScreenRenderSettingWidget::sendCameraPosToVis(const std::vector<Camera> &cameras)
{
    std::vector<CameraPoint> cameras_pos;
    cameras_pos.reserve(cameras.size());
    for(auto& camera:cameras){
        cameras_pos.emplace_back(camera.pos);
    }
    this->camera_vis_widget->SetCameraPoints(std::move(cameras_pos));
    repaint();
}
void OffScreenRenderSettingWidget::deleteCamerasItem(const std::string &name)
{
    try{
        camera_map.erase(name);
    }
    catch (const std::exception& err)
    {
        LOG_ERROR("deleteCamerasItem error: {}",err.what());
    }
}
void OffScreenRenderSettingWidget::smoothCamerasItem(const std::string &)
{
    try{

    }
    catch (const std::exception& err)
    {
        LOG_ERROR("smoothCamerasItem error: {}",err.what());
    }
}
struct RenderConfig{
    int fps;
    std::string backend;//cpu or cuda
    int iGPU;
    int width;
    int height;
    std::string output_video_name;
    bool save_image;
    std::string volume_data_config_file;
    float space_x,space_y,space_z;
    double lod_policy[10];
    vs::TransferFunc tf;
    std::string camera_sequence_config;
    std::string image_save_path;
};
bool OffScreenRenderSettingWidget::saveOffScreenRenderSettingToFile(const std::string &path)
{
    try{
        RenderConfig render_config;
        int fps = render_fps_sb->value();
        std::string backend = "cuda";//default
        int iGPU= 0;
        int width = render_width_sb->value();
        int height = render_height_sb->value();
        std::string output_video_name = output_video_name_le->text().toStdString();
        if(output_video_name.empty()) throw std::invalid_argument("output video name is empty");
        bool save_image = false;//default
        std::string image_save_path = ".";
        float space_x = 0.f,space_y = 0.f,space_z = 0.f;//these are set in volume_data_config_file
        auto render_policy = getCurrentRenderPolicy();
        auto cameras_name = render_cameras_cb->currentText().toStdString();
        std::string cameras_sequence_config = cameras_name+".json";
        if(camera_map.find(cameras_name)==camera_map.end()) throw std::invalid_argument("invalid cameras sequence");
        SaveCameraSequenceToFile(cameras_sequence_config,camera_map[cameras_name]);
        auto tf = this->tf_handle();
        if(comp_volume_config.empty()) throw std::invalid_argument("invalid volume config");
        std::string volume_data_config_file = this->comp_volume_config;

        {
            render_config.fps = fps;
            render_config.backend = backend;
            render_config.width = width;
            render_config.height = height;
            render_config.output_video_name = output_video_name;
            render_config.save_image = save_image;
            render_config.image_save_path = image_save_path;
            render_config.space_x = space_x;
            render_config.space_y = space_y;
            render_config.space_z = space_z;
            render_config.camera_sequence_config = cameras_sequence_config;
            render_config.tf = tf;
            render_config.volume_data_config_file = volume_data_config_file;
            for(int i =0;i<10;i++) render_config.lod_policy[i] = render_policy[i];
        }

        QMessageBox::information(this,
                                 "Save successfully",
                                 QString("Successfully save off-screen render setting to file: %1")
                                 .arg(path.c_str()));
        return true;
    }
    catch (const std::exception& err)
    {
        QMessageBox::warning(this,
                             "Save failed",
                             "Save off-screen render setting to file failed! Please check if all parameters are set and valid");
        LOG_ERROR("Save render setting to file failed!");
        return false;
    }
}
#define OffScreenVolumeRendererName "OffScreenVolumeRenderGUI.exe"
#define OffScreenVolumeRendererTmpConfigFile "tmp-offscreen-render-config.json"
void OffScreenRenderSettingWidget::startRenderProgram()
{
    if(!saveOffScreenRenderSettingToFile(OffScreenVolumeRendererTmpConfigFile)) return;

    bool s = QProcess::startDetached(OffScreenVolumeRendererName,{OffScreenVolumeRendererTmpConfigFile},".");

    if(s){
        LOG_INFO("Started off-screen render program");
        QMessageBox::information(this,"Start off-screen render program successfully",
                                 QString("See new program %1 for detail render information")
                                 .arg(OffScreenVolumeRendererName));
    }
    else{
        QMessageBox::warning(this,"Start off-screen render program failed",
                             QString("Please check if there is program named %1 in the same directory")
                                 .arg(OffScreenVolumeRendererName));
    }
}
auto OffScreenRenderSettingWidget::getCurrentRenderPolicy() -> std::array<double, 10>
{
    std::array<double,10> lod_policy;

    int q = render_policy_slider->value();
    for(int i=0;i<9;i++){
        lod_policy[i] = low_render_policy[i] * q;
    }

    return lod_policy;
}
void OffScreenRenderSettingWidget::setTranferFuncHandle(const OffScreenRenderSettingWidget::Handle &handle)
{
    this->tf_handle = handle;
}
void OffScreenRenderSettingWidget::setLoadedVolumeFile(const std::string &file)
{
    this->comp_volume_config = file;
}

auto LoadCameraSequenceFromFile(const std::string& camera_file)->std::vector<Camera>
{
    nlohmann::json j;

    std::ifstream in;
    in.open(camera_file);
    if(!in.is_open()){
        LOG_ERROR("Open camera sequence config file failed");
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
        LOG_ERROR("Not provide frame_count, use default frame_count(0)");
    }
    if(j.find("property") != j.end()){
        auto property = j.at("property");
        bool b0 = property[0] == "zoom";
        bool b1 = property[1] == "pos";
        bool b2 = property[2] == "look_at";
        bool b3 = property[3] == "up";
        bool b4 = property[4] == "right";
        if(!b0 || !b1 || !b2 || !b3 || !b4){
            LOG_ERROR("Camera property order or name not correct");
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

void SaveCameraSequenceToFile(const std::string& filename,const std::vector<Camera>& cameras)
{


}

void SaveRenderConfigToFile(const std::string &, const RenderConfig &)
{

}
