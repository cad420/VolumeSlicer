

set(EIGEN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen)

add_subdirectory(src)

file(
        GLOB
        ICONS
        "icons/*.png"
)

file(COPY ${ICONS}
        DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/icons/
        )