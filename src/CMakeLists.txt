list(
        APPEND
        VS_SRCS
        Slice/slice_impl.cpp
        Volume/volume_impl.cpp
        Render/render_impl.cpp
)
add_library(VS_CORE ${VS_SRCS})


set_target_properties(VS_CORE PROPERTIES PUBLIC_HEAD
        ${PROJECT_SOURCE_DIR}/include)

target_include_directories(VS_CORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/build)

install(
        TARGETS VS_CORE
        EXPORT vs-targets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include
)