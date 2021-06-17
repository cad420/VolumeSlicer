#version 430 core
in vec3 world_pos;
out vec4 frag_color;
void main()
{
    frag_color=vec4(world_pos,gl_FragCoord.w);
}
