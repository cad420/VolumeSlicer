//
// Created by wyz on 2021/4/26.
//

#ifndef NEURONANNOTATION_WGL_WRAP_HPP
#define NEURONANNOTATION_WGL_WRAP_HPP
#include <glad/wgl.h>
#include <windows.h>
#include <stdexcept>
#include <iostream>
typedef HGLRC WINAPI wglCreateContextAttribsARB_type(HDC hdc, HGLRC hShareContext,
                                                     const int *attribList);
wglCreateContextAttribsARB_type *wglCreateContextAttribsARB;

// See https://www.opengl.org/registry/specs/ARB/wgl_create_context.txt for all values
#define WGL_CONTEXT_MAJOR_VERSION_ARB             0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB             0x2092
#define WGL_CONTEXT_PROFILE_MASK_ARB              0x9126

#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB          0x00000001

typedef BOOL WINAPI wglChoosePixelFormatARB_type(HDC hdc, const int *piAttribIList,
                                                 const FLOAT *pfAttribFList, UINT nMaxFormats, int *piFormats, UINT *nNumFormats);
wglChoosePixelFormatARB_type *wglChoosePixelFormatARB;

// See https://www.opengl.org/registry/specs/ARB/wgl_pixel_format.txt for all values
#define WGL_DRAW_TO_WINDOW_ARB                    0x2001
#define WGL_ACCELERATION_ARB                      0x2003
#define WGL_SUPPORT_OPENGL_ARB                    0x2010
#define WGL_DOUBLE_BUFFER_ARB                     0x2011
#define WGL_PIXEL_TYPE_ARB                        0x2013
#define WGL_COLOR_BITS_ARB                        0x2014
#define WGL_DEPTH_BITS_ARB                        0x2022
#define WGL_STENCIL_BITS_ARB                      0x2023

#define WGL_FULL_ACCELERATION_ARB                 0x2027
#define WGL_TYPE_RGBA_ARB                         0x202B

static void
init_opengl_extensions(void)
{
    // Before we can load extensions, we need a dummy OpenGL context, created using a dummy window.
    // We use a dummy window because you can only set the pixel format for a window once. For the
    // real window, we want to use wglChoosePixelFormatARB (so we can potentially specify options
    // that aren't available in PIXELFORMATDESCRIPTOR), but we can't load and use that before we
    // have a context.
    WNDCLASSA window_class = {
            CS_HREDRAW | CS_VREDRAW | CS_OWNDC,
             DefWindowProcA,
             0,
             0,
            GetModuleHandle(0),
            0,
            0,
            0,
            0,
             "Dummy_WGL_djuasiodwa",
    };

    if (!RegisterClassA(&window_class)) {
        throw std::runtime_error("Failed to register dummy OpenGL window.");
    }

    HWND dummy_window = CreateWindowExA(
            0,
            window_class.lpszClassName,
            "Dummy OpenGL Window",
            0,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            0,
            0,
            window_class.hInstance,
            0);

    if (!dummy_window) {
        throw std::runtime_error("Failed to create dummy OpenGL window.");
    }

    HDC dummy_dc = GetDC(dummy_window);

    PIXELFORMATDESCRIPTOR pfd = {
            sizeof(pfd),
             1,
             PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
             PFD_TYPE_RGBA,

            32,
            0,
            0,
            0,
            0,
            0,
            0,
            8,
            0,
            0,
            0,
            0,
            0,
            0,
            24,
             8,
            0,
             PFD_MAIN_PLANE,
            0,
            0,
            0,
            0,
    };

    int pixel_format = ChoosePixelFormat(dummy_dc, &pfd);
    if (!pixel_format) {
        throw std::runtime_error("Failed to find a suitable pixel format.");
    }
    if (!SetPixelFormat(dummy_dc, pixel_format, &pfd)) {
        throw std::runtime_error("Failed to set the pixel format.");
    }

    HGLRC dummy_context = wglCreateContext(dummy_dc);

    if (!dummy_context) {
        throw std::runtime_error("Failed to create a dummy OpenGL rendering context.");
    }

    if (!wglMakeCurrent(dummy_dc, dummy_context)) {
        throw std::runtime_error("Failed to activate dummy OpenGL rendering context.");
    }

    wglCreateContextAttribsARB = (wglCreateContextAttribsARB_type*)wglGetProcAddress(
            "wglCreateContextAttribsARB");
    wglChoosePixelFormatARB = (wglChoosePixelFormatARB_type*)wglGetProcAddress(
            "wglChoosePixelFormatARB");

    wglMakeCurrent(dummy_dc, 0);
    wglDeleteContext(dummy_context);
    ReleaseDC(dummy_window, dummy_dc);
    DestroyWindow(dummy_window);

    UnregisterClassA(window_class.lpszClassName,window_class.hInstance);
}

static HGLRC
create_opengl_context(HDC real_dc)
{
    init_opengl_extensions();

    // Now we can choose a pixel format the modern way, using wglChoosePixelFormatARB.
    int pixel_format_attribs[] = {
            WGL_DRAW_TO_WINDOW_ARB,     GL_TRUE,
            WGL_SUPPORT_OPENGL_ARB,     GL_TRUE,
            WGL_DOUBLE_BUFFER_ARB,      GL_TRUE,
            WGL_ACCELERATION_ARB,       WGL_FULL_ACCELERATION_ARB,
            WGL_PIXEL_TYPE_ARB,         WGL_TYPE_RGBA_ARB,
            WGL_COLOR_BITS_ARB,         32,
            WGL_DEPTH_BITS_ARB,         24,
            WGL_STENCIL_BITS_ARB,       8,
            0
    };

    int pixel_format;
    UINT num_formats;
    wglChoosePixelFormatARB(real_dc, pixel_format_attribs, 0, 1, &pixel_format, &num_formats);
    if (!num_formats) {
        throw std::runtime_error("Failed to set the OpenGL pixel format.");
    }

    PIXELFORMATDESCRIPTOR pfd;
    DescribePixelFormat(real_dc, pixel_format, sizeof(pfd), &pfd);
    if (!SetPixelFormat(real_dc, pixel_format, &pfd)) {
        throw std::runtime_error("Failed to set the OpenGL pixel format.");
    }

    gladLoaderLoadWGL(real_dc);
    // Specify that we want to create an OpenGL 4.6 core profile context
    int gl_attribs[] = {
            WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
            WGL_CONTEXT_MINOR_VERSION_ARB, 6,
            WGL_CONTEXT_PROFILE_MASK_ARB,  WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
            0,
    };

    HGLRC gl_context = wglCreateContextAttribsARB(real_dc, 0, gl_attribs);

    if (!gl_context) {
        throw std::runtime_error("Failed to create OpenGL 4.6 context.");
    }

    if (!wglMakeCurrent(real_dc, gl_context)) {
        throw std::runtime_error("Failed to activate OpenGL 4.6 rendering context.");
    }

    if (!gladLoaderLoadGL()) {
        wglMakeCurrent(NULL, NULL);
        wglDeleteContext(gl_context);
    }

    return gl_context;
}

static LRESULT CALLBACK
window_callback(HWND window, UINT msg, WPARAM wparam, LPARAM lparam)
{
    LRESULT result = 0;

    switch (msg) {
        case WM_CLOSE:
        case WM_DESTROY:
            PostQuitMessage(0);
            break;
        default:
            result = DefWindowProcA(window, msg, wparam, lparam);
            break;
    }
    return result;
}

static HWND
create_window(HINSTANCE inst,LPCSTR name,int width,int height)
{
    WNDCLASSA window_class = {
             CS_HREDRAW | CS_VREDRAW | CS_OWNDC,
            window_callback,
            0,
            0,
            inst,
            0,
            LoadCursor(0, IDC_ARROW),
             0,
             0,
             name,
    };

    if (!RegisterClassA(&window_class)) {
        DWORD err=GetLastError();
        std::cout<<err;
        throw std::runtime_error("Failed to register window.");
    }

    // Specify a desired width and height, then adjust the rect so the window's client area will be
    // that size.
    RECT rect = {
            0,
            0,
            width,
            height,
    };
    DWORD window_style = WS_OVERLAPPEDWINDOW;
    AdjustWindowRect(&rect, window_style, false);

    HWND window = CreateWindowExA(
            0,
            window_class.lpszClassName,
            "OpenGL",
            window_style,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            width,
            height,
            0,
            0,
            inst,
            0);

    if (!window) {
        throw std::runtime_error("Failed to create window.");
    }
    UnregisterClassA(name,inst);
    return window;
}

#endif //NEURONANNOTATION_WGL_WRAP_HPP
