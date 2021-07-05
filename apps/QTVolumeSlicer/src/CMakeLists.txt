#QT_Dir or CMAKE_PREFIX_PATH
set(CMAKE_PREFIX_PATH C:\\Qt\\6.1.0\\msvc2019_64\\lib\\cmake)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

list(
        APPEND
        QTVolumeSlicerApp_SRCS
        main.cpp
        QTVolumeSlicerApp.cpp
        VolumeRenderWidget.cpp
        SliceRenderWidget.cpp
        SliceSettingWidget.cpp
        VolumeSettingWidget.cpp
        VolumeRenderSettingWidget.cpp
        SliceZoomWidget.cpp
)
add_executable(QTVolumeSlicerApp ${QTVolumeSlicerApp_SRCS})

target_link_libraries(QTVolumeSlicerApp PRIVATE
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
        VS_CORE
        glm::glm
        )