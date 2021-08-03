#version 430 core
layout(location=0) in vec3 vertex_pos;
uniform mat4 MVPMatrix;
void main() {
    gl_Position=MVPMatrix*(vec4(vertex_pos,1.0));
}
