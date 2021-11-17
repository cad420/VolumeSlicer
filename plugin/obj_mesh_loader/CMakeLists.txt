include(${PROJECT_SOURCE_DIR}/third_party/tinyobjloader.cmake)
list(
        APPEND
        OBJ_MESH_LOADER_SRCS
        obj_mesh_loader_plugin.cpp
)
add_library(
        ObjMeshLoaderPlugin
        SHARED
        ${OBJ_MESH_LOADER_SRCS}
)
target_link_libraries(
        ObjMeshLoaderPlugin
        PRIVATE
        VS_CORE
        tinyobjloader
)
set_target_properties(
        ObjMeshLoaderPlugin PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/plugins
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/plugins
)