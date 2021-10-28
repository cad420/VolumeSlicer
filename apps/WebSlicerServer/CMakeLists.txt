include(third_party/poco.cmake)
include(third_party/seria.cmake)
list(
        APPEND
        SRCS
        src/main.cpp
        src/SlicerServerApplication.cpp
        src/Server/RequestHandlerFactory.cpp
        src/Server/WebSocketRequestHandler.cpp
)

add_executable(SlicerServer ${SRCS})
target_include_directories(
        SlicerServer
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(
        SlicerServer
        PRIVATE
        Poco::Net
        Poco::Util
        VS_CORE
        seria::seria
)