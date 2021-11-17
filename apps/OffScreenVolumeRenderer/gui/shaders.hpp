//
// Created by wyz on 2021/11/17.
//
#pragma once
namespace shader{
    inline const char* line_shader_v = R"(#version 330 core
layout(location = 0) in vec3 VertexPos;
uniform mat4 MVPMatrix;
void main() {
    gl_Position = MVPMatrix * vec4(VertexPos,1.f);
}
)";

    inline const char* line_shader_f = R"(#version 330 core
out vec4 frag_color;
uniform vec4 line_color;
void main() {
    frag_color = line_color;
}
)";

}
