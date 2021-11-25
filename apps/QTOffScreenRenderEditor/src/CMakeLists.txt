

#QT_Dir or CMAKE_PREFIX_PATH
set(CMAKE_PREFIX_PATH C:\\Qt\\6.1.0\\msvc2019_64\\lib\\cmake)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets OpenGL OpenGLWidgets)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

list(
        APPEND
        QTOffScreenRenderEditor_SRCS
        main.cpp
        QTOffScreenRenderEditor.cpp
        SettingWidget.cpp
        VolumeRenderWidget.cpp
        TransferFunctionWidget.cpp
        TFCanvas.cpp
        DoubleSlider.cpp
        tf1deditor.cpp
        tf1dmappingcanvas.cpp
        tf1dtexturecanvas.cpp
        BSplineCurve.cpp
)

add_executable(QTOffScreenRenderEditor ${QTOffScreenRenderEditor_SRCS})

target_include_directories(QTOffScreenRenderEditor
        PRIVATE
        ../third_party
        s)



target_include_directories(
        QTOffScreenRenderEditor
        PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN_INCLUDE_DIR}
)
target_link_libraries(
        QTOffScreenRenderEditor
        PRIVATE
        VS_CORE
        ${OpenCV_LIBS}
)
target_compile_features(
        QTOffScreenRenderEditor
        PRIVATE
        cxx_std_17
)

target_link_libraries(QTOffScreenRenderEditor
        PUBLIC
        VS_CORE
        PRIVATE
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
        Qt6::OpenGL
        Qt6::OpenGLWidgets
        OpenMP::OpenMP_CXX
        glm::glm
        )