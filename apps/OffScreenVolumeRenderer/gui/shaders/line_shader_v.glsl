#version 330 core
layout(location = 0) in vec3 VertexPos;
uniform mat4 MVPMatrix;
void main() {
    gl_Position = MVPMatrix * vec4(VertexPos,1.f);
}
