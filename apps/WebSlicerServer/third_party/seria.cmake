include(FetchContent)
FetchContent_Declare(
        seria
        GIT_REPOSITORY https://github.com/ukabuer/seria.git
        GIT_TAG v0.2
        GIT_SHALLOW true
        GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(seria)