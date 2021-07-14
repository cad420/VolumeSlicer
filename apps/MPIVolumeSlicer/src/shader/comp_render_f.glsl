#version 430 core
out vec4 frag_color;
layout(binding=0,r8) uniform image2D comp_sample_texture;
void main() {
    float scalar=imageLoad(comp_sample_texture,ivec2(gl_FragCoord.xy)).x;
    frag_color=vec4(scalar,scalar,scalar,1.f);
//    frag_color=vec4(1.f,0.f,0.f,1.f);
}
