include_guard()

include(FetchContent)
FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG 0.9.9.8
        GIT_SHALLOW true
        GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(glm)