
option(VS_BUILD_APP_MPILargeMeshVis "" OFF)
option(VS_BUILD_APP_MPILargeVolumeVis "" OFF)
option(VS_BUILD_APP_MPIVolumeSlicer "" OFF)
option(VS_BUILD_APP_OffScreenVolumeRenderer "" OFF)
option(VS_BUILD_APP_QTOffScreenRenderEditor "" OFF)
option(VS_BUILD_APP_QTVolumeSlicer "" OFF)
option(VS_BUILD_APP_RawVolumeRenderer "" OFF)
option(VS_BUILD_APP_WebSlicerServer "" OFF)
if(VS_BUILD_APP_MPILargeMeshVis
        OR VS_BUILD_APP_MPILargeVolumeVis
        OR VS_BUILD_APP_MPIVolumeSlicer
        OR VS_BUILD_APP_RawVolumeRenderer)
    include(${PROJECT_SOURCE_DIR}/third_party/glfw.cmake)
    include(${PROJECT_SOURCE_DIR}/third_party/sdl2.cmake)
    include(${PROJECT_SOURCE_DIR}/third_party/imgui.cmake)
    if(WIN32)
        include(${PROJECT_SOURCE_DIR}/third_party/msmpi.cmake)
    elseif(UNIX AND NOT APPLE)
        find_package(MPI)
    endif()
endif()


find_package(OpenGL REQUIRED)

if(VS_BUILD_APP_OffScreenVolumeRenderer OR VS_BUILD_APP_QTOffScreenRenderEditor)
    set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/OffScreenVolumeRenderer/third_party/opencv)
    find_package(OpenCV REQUIRED)
endif()

if(VS_BUILD_APP_QTVolumeSlicer)
    add_subdirectory(QTVolumeSlicer)
endif()

if(VS_BUILD_APP_MPIVolumeSlicer)
    add_subdirectory(MPIVolumeSlicer)
endif()

if(VS_BUILD_APP_MPILargeVolumeVis)
    add_subdirectory(MPILargeVolumeVis)
endif()

if(VS_BUILD_APP_OffScreenVolumeRenderer)
    add_subdirectory(OffScreenVolumeRenderer)
endif()

if(VS_BUILD_APP_WebSlicerServer)
    add_subdirectory(WebSlicerServer)
endif()

if(VS_BUILD_APP_MPILargeMeshVis)
    add_subdirectory(MPILargeMeshVis)
endif()

if(VS_BUILD_APP_QTOffScreenRenderEditor)
    add_subdirectory(QTOffScreenRenderEditor)
endif()