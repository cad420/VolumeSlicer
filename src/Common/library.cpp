//
// Created by wyz on 2021/9/29.
//
#include <Utils/library.hpp>
#include <Utils/logger.hpp>
#if defined(_WIN32)
#include <Windows.h>
#elif defined(__linux__)
#include <sys/times.h>
#include <dlfcn.h>
#endif

VS_START
Library::Library(const std::string &name)
:lib(nullptr)
{
    std::string error_msg;
#if defined(_WIN32)
    lib = LoadLibrary(TEXT(name.c_str()));
    if( !lib ){
        auto err = GetLastError();
        LPTSTR lpMsgBuf;
        FormatMessage(
            FORMAT_MESSAGE_ALLOCATE_BUFFER |
            FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL,
            err,
            MAKELANGID( LANG_NEUTRAL, SUBLANG_DEFAULT ),
            (LPTSTR)&lpMsgBuf,
            0, NULL );
        error_msg = lpMsgBuf;
        LocalFree(lpMsgBuf);
    }
#elif defined(__linux__)

#endif
    if(!lib){
        LOG_ERROR(name+" not found!");
        throw std::runtime_error(error_msg);
    }
}
void* Library::Symbol(const std::string &name) const
{
    assert(lib);
#if defined(_WIN32)
    return GetProcAddress(reinterpret_cast<HMODULE>(lib),name.c_str());
#elif defined(__linux__)

#endif
}
void Library::Close()
{
#if defined(_WIN32)
    FreeLibrary(reinterpret_cast<HMODULE>(lib));
#elif defined(__linux__)

#endif
}
Library::~Library()
{
    Close();
}
VS_END
