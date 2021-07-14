include(FetchContent)
FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui
        GIT_TAG v1.79
        GIT_SHALLOW true
        GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(imgui)

set(IMGUI_ROOT ${CMAKE_BINARY_DIR}/_deps/imgui-src)


aux_source_directory(${IMGUI_ROOT} IMGUI_SRC)

add_library(imgui STATIC ${IMGUI_SRC}
        ${IMGUI_ROOT}/examples/imgui_impl_sdl.cpp
        ${IMGUI_ROOT}/examples/imgui_impl_opengl3.cpp)
target_include_directories(imgui
        PUBLIC
        ${IMGUI_ROOT}
        ${IMGUI_ROOT}/examples
        PRIVATE
        ${CMAKE_BINARY_DIR}/_deps/sdl2-src/include
        ${PROJECT_SOURCE_DIR}/third_party/glad/include
        )

target_compile_definitions(imgui PRIVATE
        IMGUI_IMPL_OPENGL_LOADER_GLAD
        IMGUI_IMPL_OPENGL_LOADER_GLBINDING3)