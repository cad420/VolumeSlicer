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