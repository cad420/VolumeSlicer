//
// Created by wyz on 2021/11/5.
//
#include <iostream>
#include <string>

#ifdef GLAPI

inline std::string GetGLErrorStr(GLenum gl_error)
{
    std::string error;
    switch (gl_error)
    {
    case GL_INVALID_ENUM:
        error = "GL_INVALID_ENUM";
        break;
    case GL_INVALID_VALUE:
        error = "GL_INVALID_VALUE";
        break;
    case GL_INVALID_OPERATION:
        error = "GL_INVALID_OPERATION";
        break;
    case GL_STACK_OVERFLOW:
        error = "GL_STACK_OVERFLOW";
        break;
    case GL_STACK_UNDERFLOW:
        error = "GL_STACK_UNDERFLOW";
        break;
    case GL_OUT_OF_MEMORY:
        error = "GL_OUT_OF_MEMORY";
        break;
    case GL_INVALID_FRAMEBUFFER_OPERATION:
        error = "GL_INVALID_FRAMEBUFFER_OPERATION";
        break;
    case GL_INVALID_INDEX:
        error = "GL_INVALID_INDEX";
        break;
    default:
        error = "UNKNOWN_ERROR";
        break;
    }
    return error;
}

inline void PrintGLErrorType(GLenum gl_error)
{
    std::cout << GetGLErrorStr(gl_error) << std::endl;
}

inline GLenum PrintGLErrorMsg(const char *file, int line)
{
    GLenum error_code;
    while ((error_code = glGetError()) != GL_NO_ERROR)
    {
        std::string error;
        switch (error_code)
        {
        case GL_INVALID_ENUM:
            error = "GL_INVALID_ENUM";
            break;
        case GL_INVALID_VALUE:
            error = "GL_INVALID_VALUE";
            break;
        case GL_INVALID_OPERATION:
            error = "GL_INVALID_OPERATION";
            break;
        case GL_STACK_OVERFLOW:
            error = "GL_STACK_OVERFLOW";
            break;
        case GL_STACK_UNDERFLOW:
            error = "GL_STACK_UNDERFLOW";
            break;
        case GL_OUT_OF_MEMORY:
            error = "GL_OUT_OF_MEMORY";
            break;
        case GL_INVALID_FRAMEBUFFER_OPERATION:
            error = "GL_INVALID_FRAMEBUFFER_OPERATION";
            break;
        case GL_INVALID_INDEX:
            error = "GL_INVALID_INDEX";
            break;
        }
        std::cout << error_code << " at line " << line << " in file " << file << std::endl;
    }
    return error_code;
}

#endif

#ifdef NDEBUG
#define GL_REPORT void(0);
#define GL_ASSERT void(0);
#define GL_EXPR(expr) expr;
#define GL_CHECK void(0);
#else
#define GL_REPORT PrintGLErrorMsg(__FILE__, __LINE__);
#define GL_ASSERT assert(glGetError() == GL_NO_ERROR);

#define GL_EXPR(expr)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        GLenum gl_error;                                                                                               \
        int count = 0;                                                                                                 \
        while ((gl_error = glGetError()) != GL_NO_ERROR)                                                               \
        {                                                                                                              \
            std::cout << "GL error " << GetGLErrorStr(gl_error) << " before call " << #expr << " at line " << __LINE__ \
                      << " in file " << __FILE__ << std::endl;                                                         \
            count++;                                                                                                   \
            if (count > 10)                                                                                            \
                break;                                                                                                 \
        }                                                                                                              \
        expr;                                                                                                          \
        count = 0;                                                                                                     \
        while ((gl_error = glGetError()) != GL_NO_ERROR)                                                               \
        {                                                                                                              \
            std::cout << "Calling " << #expr << " caused GL error " << GetGLErrorStr(gl_error) << " at line "          \
                      << __LINE__ << " in file " << __FILE__ << std::endl;                                             \
            if (++count > 10)                                                                                          \
                break;                                                                                                 \
        }                                                                                                              \
    } while (false);

#define GL_CHECK                                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        GLenum gl_error;                                                                                               \
        int count = 0;                                                                                                 \
        while ((gl_error = glGetError()) != GL_NO_ERROR)                                                               \
        {                                                                                                              \
            std::cout << "GL error " << GetGLErrorStr(gl_error) << " before line " << __LINE__ << " in file "          \
                      << __FILE__ << std::endl;                                                                        \
            if (++count > 10)                                                                                          \
                break;                                                                                                 \
        }                                                                                                              \
    } while (0);

#endif
