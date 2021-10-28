set(FETCHCONTENT_QUIET FALSE)

include(FetchContent)
FetchContent_Declare(
    seria
    GIT_REPOSITORY https://github.com/ukabuer/seria.git
    GIT_TAG v0.1
    GIT_SHALLOW true
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(seria)