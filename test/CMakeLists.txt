add_executable(test_template test_template.cpp)
target_link_libraries(test_template
        PRIVATE
        VS_CORE)

add_executable(test_pyramid_intersect_aabb test_pyramid_intersect_aabb.cpp)
target_link_libraries(test_pyramid_intersect_aabb
        PRIVATE
        VS_CORE)
target_include_directories(test_pyramid_intersect_aabb
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src)

add_executable(test_texture test_texture.cpp)
target_link_libraries(test_texture
        PRIVATE
        VS_CORE)
target_include_directories(test_texture
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src)

add_executable(test_raw_cpu_volume_render test_raw_cpu_volume_render.cpp)
target_link_libraries(test_raw_cpu_volume_render
        PRIVATE
        VS_CORE)
target_include_directories(test_raw_cpu_volume_render
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src)

add_executable(test_comp_cpu_volume_render test_comp_cpu_volume_render.cpp)
target_link_libraries(test_comp_cpu_volume_render
        PRIVATE
        VS_CORE)
target_include_directories(test_comp_cpu_volume_render
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src)

add_executable(test_comp_cuda_volume_render test_comp_cuda_volume_render.cpp)
target_link_libraries(test_comp_cuda_volume_render
        PRIVATE
        VS_CORE)
target_include_directories(test_comp_cuda_volume_render
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src)