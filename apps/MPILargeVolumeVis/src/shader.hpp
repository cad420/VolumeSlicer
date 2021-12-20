#pragma once

namespace shader{

    inline const char* composite_render_v = R"(#version 430 core
layout(location=0) in vec2 vertex_pos;
void main() {
    gl_Position=vec4(vertex_pos,0.0f,1.0f);
}
)";

    inline const char* conposite_render_f = R"(#version 430 core
layout(location = 0) out vec4 outFragColor;
layout(binding = 0,rgba8) uniform image2D CompRenderTex;

void main() {
    outFragColor = imageLoad(CompRenderTex,ivec2(gl_FragCoord.x,gl_FragCoord.y)).rgba;
}
)";
}
