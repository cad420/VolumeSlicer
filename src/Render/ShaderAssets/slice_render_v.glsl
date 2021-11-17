#version 430 core
layout(location=0) in vec3 vertex_pos;
uniform mat4 MVPMatrix;
out vec3 world_pos;
void main() {
    gl_Position=MVPMatrix*(vec4(vertex_pos,1.0));
    world_pos=vertex_pos;
}
