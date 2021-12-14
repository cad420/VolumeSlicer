
option(BUILD_TEST "whether build test" OFF)
option(BUILD_TOOL "whether build tool" OFF)

include(FetchContent)
FetchContent_Declare(
        VolumeCompression
        GIT_REPOSITORY https://github.com/wyzwzz/VolumeCompression.git
        GIT_TAG master
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(VolumeCompression)