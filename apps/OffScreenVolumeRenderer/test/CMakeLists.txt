

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
