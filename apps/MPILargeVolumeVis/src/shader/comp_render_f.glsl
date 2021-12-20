#version 430 core
layout(location = 0) out vec4 outFragColor;
layout(binding = 0,rgba8) uniform image2D CompRenderTex;

void main() {
    outFragColor = imageLoad(CompRenderTex,ivec2(gl_FragCoord.x,gl_FragCoord.y)).rgba;
}
