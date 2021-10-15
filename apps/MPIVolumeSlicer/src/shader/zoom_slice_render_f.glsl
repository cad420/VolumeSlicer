#version 430 core
out vec4 frag_color;
layout(binding=0,r8) uniform image2D comp_sample_texture;
uniform int min_p_x;
uniform int max_p_x;
uniform int min_p_y;
uniform int max_p_y;
void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    if(((coord.x==min_p_x || coord.x==max_p_x) && coord.y>=min_p_y && coord.y<=max_p_y)
    || ((coord.y==min_p_y || coord.y==max_p_y) && coord.x>=min_p_x && coord.x<=max_p_x)){
        frag_color=vec4(1.f,0.f,0.f,1.f);
    }
    else{
        float scalar=imageLoad(comp_sample_texture,coord).x;
        frag_color=vec4(scalar,scalar,scalar,1.f);
    }
}
