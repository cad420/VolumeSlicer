

add_executable(test_offscreen_render_tovideo
        test_offscreen_render_tovideo.cpp
        ../src/VideoCapture.cpp)
target_link_libraries(
        test_offscreen_render_tovideo
        PRIVATE
        VS_CORE
        ${OpenCV_LIBS}
)
target_include_directories(
        test_offscreen_render_tovideo
        PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ../src
)

add_executable(test_camera_sequence test_camera_sequence.cpp)
target_include_directories(test_camera_sequence
        PRIVATE
        ${PROJECT_SOURCE_DIR}/third_party
        )

add_executable(test_image_file_toviode
        test_image_file_toviode.cpp
        ../src/VideoCapture.cpp)
target_link_libraries(
        test_image_file_toviode
        PRIVATE
        VS_CORE
        ${OpenCV_LIBS}
)
target_include_directories(
        test_image_file_toviode
        PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ../src
)