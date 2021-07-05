#version 430 core
in vec3 world_pos;
out vec4 frag_color;
bool render_board;
void main()
{
    if(render_board)
        frag_color=vec4(1.f,0.f,1.f,0.f);//pink
    else
        frag_color=vec4(world_pos,gl_FragCoord.w);
}
