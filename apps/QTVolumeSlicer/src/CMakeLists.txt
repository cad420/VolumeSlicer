#QT_Dir or CMAKE_PREFIX_PATH
set(CMAKE_PREFIX_PATH C:\\Qt\\6.1.0\\msvc2019_64\\lib\\cmake)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets OpenGL OpenGLWidgets)
find_package(OpenMP REQUIRED)
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
        TransferFunctionWidget.cpp
        TFCanvas.cpp
        DoubleSlider.cpp
        tf1deditor.cpp
        tf1dmappingcanvas.cpp
        tf1dtexturecanvas.cpp
)
add_executable(QTVolumeSlicerApp ${QTVolumeSlicerApp_SRCS})

target_link_libraries(QTVolumeSlicerApp PRIVATE
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
        Qt6::OpenGL
        Qt6::OpenGLWidgets
        OpenMP::OpenMP_CXX
        VS_CORE
        glm::glm
        )